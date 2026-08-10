#!/usr/bin/env python3
"""
实验和基准测试模块
用于验证算法正确性和评估性能
"""

from typing import Dict
import time
import json
import os

from .idiom_data import IdiomDictionary, create_sample_data
from .idiom_graph import IdiomGraph, GameState
from .sg_solver import SGSolver, TailGroupedSolver, simulate_game
from .minimax_solver import MinimaxSolver


class ExperimentRunner:
    """
    实验运行器
    执行各种实验并生成报告
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化实验运行器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results: Dict = {}
    
    def run_small_scale_test(self, n_idioms: int = 20) -> Dict:
        """
        小规模测试（验证算法正确性）
        
        Args:
            n_idioms: 成语数量
        
        Returns:
            测试结果
        """
        print(f"\n=== 小规模测试 (n={n_idioms}) ===")
        
        # 加载数据
        dict_obj = IdiomDictionary(use_pinyin=True)
        dict_obj.load_from_list(create_sample_data(n_idioms))
        
        graph = IdiomGraph(dict_obj)
        
        # SG 求解器测试
        sg_solver = SGSolver(graph)
        initial_sg = sg_solver.calculate_sg_initial()
        
        # Minimax 求解器测试
        mm_solver = MinimaxSolver(graph)
        initial_state = GameState()
        is_win_mm, mm_value = mm_solver.is_winning(initial_state)
        
        # 验证一致性
        sg_winning = initial_sg > 0
        mm_winning = is_win_mm
        
        consistency = (sg_winning == mm_winning)
        
        # 模拟游戏
        game_seq, first_wins = simulate_game(sg_solver, graph, max_moves=50)
        
        result = {
            'n_idioms': n_idioms,
            'graph_stats': graph.get_stats(),
            'sg_results': {
                'initial_sg': initial_sg,
                'is_winning': sg_winning,
                'cache_stats': sg_solver.get_cache_stats(),
            },
            'minimax_results': {
                'game_value': mm_value,
                'is_winning': mm_winning,
                'stats': mm_solver.get_stats(),
            },
            'consistency_check': {
                'sg_and_minimax_agree': consistency,
                'expected': True,
                'passed': consistency,
            },
            'simulation': {
                'game_length': len(game_seq),
                'first_player_wins': first_wins,
                'sequence': [dict_obj.get_text(id_) for id_ in game_seq[:10]],
            },
        }
        
        print(f"图统计: {graph.get_stats()}")
        print(f"SG 初值: {initial_sg}, 必胜: {sg_winning}")
        print(f"Minimax 值: {mm_value}, 必胜: {mm_winning}")
        print(f"一致性检查: {'通过' if consistency else '失败'}")
        print(f"模拟游戏长度: {len(game_seq)}, 先手胜: {first_wins}")
        
        self.results['small_scale'] = result
        return result
    
    def run_medium_scale_test(self, n_idioms: int = 50) -> Dict:
        """
        中规模测试（性能评估）
        
        Args:
            n_idioms: 成语数量
        
        Returns:
            测试结果
        """
        print(f"\n=== 中规模测试 (n={n_idioms}) ===")
        
        # 加载数据
        dict_obj = IdiomDictionary(use_pinyin=True)
        dict_obj.load_from_list(create_sample_data(n_idioms))
        
        graph = IdiomGraph(dict_obj)
        
        # SG 求解器
        start_time = time.time()
        sg_solver = SGSolver(graph)
        initial_sg = sg_solver.calculate_sg_initial()
        sg_time = time.time() - start_time
        
        # Minimax 求解器
        start_time = time.time()
        mm_solver = MinimaxSolver(graph)
        is_win_mm, mm_value = mm_solver.is_winning(GameState())
        mm_time = time.time() - start_time
        
        # 分组求解器
        grouped_solver = TailGroupedSolver(graph)
        group_stats = grouped_solver.get_group_stats()
        
        result = {
            'n_idioms': n_idioms,
            'graph_stats': graph.get_stats(),
            'sg_solver': {
                'initial_sg': initial_sg,
                'time_seconds': round(sg_time, 3),
                'cache_stats': sg_solver.get_cache_stats(),
            },
            'minimax_solver': {
                'game_value': mm_value,
                'time_seconds': round(mm_time, 3),
                'stats': mm_solver.get_stats(),
            },
            'grouped_solver': {
                'group_stats': group_stats,
            },
            'performance_comparison': {
                'sg_time': sg_time,
                'minimax_time': mm_time,
                'sg_faster': sg_time < mm_time,
            },
        }
        
        print(f"SG 求解时间: {sg_time:.3f}s")
        print(f"Minimax 求解时间: {mm_time:.3f}s")
        print(f"分组统计: {group_stats}")
        
        self.results['medium_scale'] = result
        return result
    
    def run_analyze_all_moves(self, n_idioms: int = 30) -> Dict:
        """
        分析所有开局成语
        
        Args:
            n_idioms: 成语数量
        
        Returns:
            分析结果
        """
        print(f"\n=== 开局分析 (n={n_idioms}) ===")
        
        dict_obj = IdiomDictionary(use_pinyin=True)
        dict_obj.load_from_list(create_sample_data(n_idioms))
        
        graph = IdiomGraph(dict_obj)
        solver = SGSolver(graph)
        
        # 分析所有开局
        analysis = solver.analyze_all_initial_moves()
        
        # 分类统计
        winning_starts = []
        losing_starts = []
        
        for idiom_id, (text, sg, is_good) in analysis.items():
            if is_good:
                winning_starts.append((text, sg))
            else:
                losing_starts.append((text, sg))
        
        # 排序
        winning_starts.sort(key=lambda x: -x[1])
        losing_starts.sort(key=lambda x: x[1])
        
        result = {
            'n_idioms': n_idioms,
            'total_starts': len(analysis),
            'winning_starts_count': len(winning_starts),
            'losing_starts_count': len(losing_starts),
            'winning_starts_top5': winning_starts[:5],
            'losing_starts_top5': losing_starts[:5],
            'win_rate': len(winning_starts) / len(analysis) if analysis else 0,
        }
        
        print(f"总开局数: {len(analysis)}")
        print(f"必胜开局: {len(winning_starts)} ({result['win_rate']*100:.1f}%)")
        print(f"必败开局: {len(losing_starts)}")
        print(f"最佳开局: {winning_starts[:5]}")
        print(f"最差开局: {losing_starts[:5]}")
        
        self.results['opening_analysis'] = result
        return result
    
    def run_correctness_verification(self) -> Dict:
        """
        正确性验证测试
        
        Returns:
            验证结果
        """
        print(f"\n=== 正确性验证 ===")
        
        tests_passed = []
        tests_failed = []
        
        # 测试 1: SG 值定义一致性
        print("测试 1: SG 值定义一致性...")
        dict_obj = IdiomDictionary()
        dict_obj.load_from_list(create_sample_data(10))
        graph = IdiomGraph(dict_obj)
        solver = SGSolver(graph)
        
        # 验证：终止状态 SG = 0
        dead_ends = graph.find_dead_ends()
        for dead_id in dead_ends:
            state = GameState(last_idiom=dead_id, used_set={dead_id})
            sg = solver.calculate_sg(state)
            if sg == 0:
                tests_passed.append("dead_end_sg_0")
            else:
                tests_failed.append(f"dead_end_sg_not_0: {dead_id}")
        
        # 测试 2: Minimax 和 SG 一致性
        print("测试 2: Minimax 和 SG 一致性...")
        mm_solver = MinimaxSolver(graph)
        
        for n_idioms in [10, 20, 30]:
            dict_obj = IdiomDictionary()
            dict_obj.load_from_list(create_sample_data(n_idioms))
            graph = IdiomGraph(dict_obj)
            
            sg_solver = SGSolver(graph)
            mm_solver = MinimaxSolver(graph)
            
            sg_win = sg_solver.calculate_sg_initial() > 0
            mm_win = mm_solver.is_winning(GameState())[0]
            
            if sg_win == mm_win:
                tests_passed.append(f"consistency_n_{n_idioms}")
            else:
                tests_failed.append(f"inconsistency_n_{n_idioms}")
        
        # 测试 3: 必胜策略有效性
        print("测试 3: 必胜策略有效性...")
        dict_obj = IdiomDictionary()
        dict_obj.load_from_list(create_sample_data(20))
        graph = IdiomGraph(dict_obj)
        solver = SGSolver(graph)
        
        initial_sg = solver.calculate_sg_initial()
        if initial_sg > 0:
            # 验证必胜策略
            best_move = solver.find_best_move(GameState())
            if best_move:
                new_state = GameState().make_move(best_move[0])
                new_sg = solver.calculate_sg(new_state)
                if new_sg == 0:
                    tests_passed.append("winning_strategy_correct")
                else:
                    tests_failed.append("winning_strategy_wrong")
        
        result = {
            'tests_passed': tests_passed,
            'tests_failed': tests_failed,
            'total_tests': len(tests_passed) + len(tests_failed),
            'pass_rate': len(tests_passed) / (len(tests_passed) + len(tests_failed))
                         if (len(tests_passed) + len(tests_failed)) > 0 else 0,
            'all_passed': len(tests_failed) == 0,
        }
        
        print(f"通过测试: {len(tests_passed)}")
        print(f"失败测试: {len(tests_failed)}")
        print(f"通过率: {result['pass_rate']*100:.1f}%")
        
        self.results['correctness'] = result
        return result
    
    def run_all_experiments(self) -> Dict:
        """
        运行所有实验
        
        Returns:
            所有实验结果
        """
        print("\n" + "="*50)
        print("成语接龙博弈求解器 - 完整实验")
        print("="*50)
        
        self.run_small_scale_test(20)
        self.run_medium_scale_test(50)
        self.run_analyze_all_moves(30)
        self.run_correctness_verification()
        
        return self.results
    
    def save_results(self, filename: str = "experiment_results.json") -> str:
        """
        保存实验结果
        
        Args:
            filename: 文件名
        
        Returns:
            文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """
        生成实验报告
        
        Returns:
            报告文本
        """
        report = []
        report.append("="*60)
        report.append("成语接龙博弈求解器 - 实验报告")
        report.append("="*60)
        
        if 'small_scale' in self.results:
            r = self.results['small_scale']
            report.append("\n## 小规模测试结果")
            report.append(f"- 成语数量: {r['n_idioms']}")
            report.append(f"- 图节点数: {r['graph_stats']['num_nodes']}")
            report.append(f"- 图边数: {r['graph_stats']['num_edges']}")
            report.append(f"- 初始 SG 值: {r['sg_results']['initial_sg']}")
            report.append(f"- 先手必胜: {r['sg_results']['is_winning']}")
            report.append(f"- 一致性检查: {'通过' if r['consistency_check']['passed'] else '失败'}")
        
        if 'medium_scale' in self.results:
            r = self.results['medium_scale']
            report.append("\n## 中规模测试结果")
            report.append(f"- 成语数量: {r['n_idioms']}")
            report.append(f"- SG 求解时间: {r['sg_solver']['time_seconds']}s")
            report.append(f"- Minimax 求解时间: {r['minimax_solver']['time_seconds']}s")
            report.append(f"- 缓存命中率: {r['sg_solver']['cache_stats']['hit_rate']}")
        
        if 'opening_analysis' in self.results:
            r = self.results['opening_analysis']
            report.append("\n## 开局分析结果")
            report.append(f"- 总开局数: {r['total_starts']}")
            report.append(f"- 必胜开局数: {r['winning_starts_count']}")
            report.append(f"- 必败开局数: {r['losing_starts_count']}")
            report.append(f"- 先手胜率: {r['win_rate']*100:.1f}%")
        
        if 'correctness' in self.results:
            r = self.results['correctness']
            report.append("\n## 正确性验证")
            report.append(f"- 通过测试数: {len(r['tests_passed'])}")
            report.append(f"- 失败测试数: {len(r['tests_failed'])}")
            report.append(f"- 通过率: {r['pass_rate']*100:.1f}%")
            report.append(f"- 全部通过: {'是' if r['all_passed'] else '否'}")
        
        return "\n".join(report)


def main():
    """主函数"""
    runner = ExperimentRunner()
    runner.run_all_experiments()
    runner.save_results()
    
    report = runner.generate_report()
    print("\n" + report)
    
    # 保存报告
    with open("results/experiment_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    main()
