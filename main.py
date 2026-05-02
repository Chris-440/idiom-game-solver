#!/usr/bin/env python3
"""
成语接龙博弈求解器 - 主程序入口
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.idiom_data import IdiomDictionary, create_sample_data
from src.idiom_graph import IdiomGraph, GameState
from src.sg_solver import SGSolver, simulate_game
from src.minimax_solver import MinimaxSolver
from src.experiment import ExperimentRunner


def demo_basic():
    """基础演示"""
    print("\n" + "="*50)
    print("成语接龙博弈求解器 - 基础演示")
    print("="*50)
    
    # 加载示例数据
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(30))
    
    print(f"\n加载成语数量: {len(dict_obj)}")
    print(f"统计信息: {dict_obj.get_stats()}")
    
    # 构建图
    graph = IdiomGraph(dict_obj)
    print(f"\n图统计: {graph.get_stats()}")
    
    # SG 求解
    solver = SGSolver(graph)
    initial_sg = solver.calculate_sg_initial()
    
    print(f"\n=== SG 求解结果 ===")
    print(f"初始状态 SG 值: {initial_sg}")
    print(f"先手必胜: {initial_sg > 0}")
    
    # 找最优开局
    best_move = solver.find_best_move(GameState())
    if best_move:
        print(f"最优开局: {best_move[1]}")
    
    # 缓存统计
    print(f"\n缓存统计: {solver.get_cache_stats()}")
    
    # 模拟游戏
    print(f"\n=== 模拟游戏 ===")
    seq, first_wins = simulate_game(solver, graph, max_moves=20)
    print(f"游戏序列: {[dict_obj.get_text(id_) for id_ in seq]}")
    print(f"游戏长度: {len(seq)}")
    print(f"先手获胜: {first_wins}")


def demo_minimax():
    """Minimax 演示"""
    print("\n" + "="*50)
    print("Minimax + Alpha-Beta 剪枝演示")
    print("="*50)
    
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(25))
    
    graph = IdiomGraph(dict_obj)
    solver = MinimaxSolver(graph)
    
    # 求解
    is_win, value = solver.is_winning(GameState())
    
    print(f"\n博弈值: {value}")
    print(f"先手必胜: {is_win}")
    
    # 找最优移动
    best = solver.find_best_move(GameState())
    if best:
        print(f"最优开局: {best[1]}")
    
    # 分析局面
    analysis = solver.analyze_position(GameState())
    print(f"\n合法移动数: {analysis['total_moves']}")
    
    if analysis['legal_moves']:
        print("前5个移动分析:")
        for move in analysis['legal_moves'][:5]:
            print(f"  - {move['text']}: 值={move['value']}, 必胜={move['is_winning_for_current']}")
    
    # 统计
    print(f"\n求解统计: {solver.get_stats()}")


def demo_analysis():
    """开局分析演示"""
    print("\n" + "="*50)
    print("开局成语分析")
    print("="*50)
    
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(40))
    
    graph = IdiomGraph(dict_obj)
    solver = SGSolver(graph)
    
    # 分析所有开局
    analysis = solver.analyze_all_initial_moves()
    
    # 分类
    winning = [(text, sg) for _, (text, sg, is_good) in analysis.items() if is_good]
    losing = [(text, sg) for _, (text, sg, is_good) in analysis.items() if not is_good]
    
    winning.sort(key=lambda x: -x[1])
    losing.sort(key=lambda x: x[1])
    
    print(f"\n总开局数: {len(analysis)}")
    print(f"必胜开局: {len(winning)} ({len(winning)/len(analysis)*100:.1f}%)")
    print(f"必败开局: {len(losing)} ({len(losing)/len(analysis)*100:.1f}%)")
    
    print(f"\n最佳开局 (对手 SG 值最低):")
    for text, sg in winning[:5]:
        print(f"  - {text}: SG={sg}")
    
    print(f"\n最差开局 (对手 SG 值最高):")
    for text, sg in losing[:5]:
        print(f"  - {text}: SG={sg}")


def run_full_experiment():
    """运行完整实验"""
    print("\n" + "="*50)
    print("运行完整实验")
    print("="*50)
    
    runner = ExperimentRunner()
    runner.run_all_experiments()
    runner.save_results()
    
    report = runner.generate_report()
    print(report)


def interactive_game():
    """交互式游戏"""
    print("\n" + "="*50)
    print("交互式成语接龙")
    print("="*50)
    
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(50))
    
    graph = IdiomGraph(dict_obj)
    solver = SGSolver(graph)
    
    state = GameState()
    
    print("\n游戏开始！你可以输入成语，AI 会给出最优回应。")
    print("输入 'quit' 退出，输入 'hint' 获取提示。")
    
    while True:
        # 显示当前状态
        if state.last_idiom:
            print(f"\n上一个成语: {dict_obj.get_text(state.last_idiom)}")
        else:
            print("\n游戏开始，请输入第一个成语。")
        
        # 用户输入
        user_input = input("你的成语: ").strip()
        
        if user_input == 'quit':
            print("游戏结束。")
            break
        
        if user_input == 'hint':
            best = solver.find_best_move(state)
            if best:
                print(f"提示: 建议使用 '{best[1]}'")
            else:
                print("提示: 没有合法移动了！")
            continue
        
        # 查找成语
        found_id = None
        for id_ in dict_obj.get_all_ids():
            if dict_obj.get_text(id_) == user_input:
                found_id = id_
                break
        
        if found_id is None:
            print(f"'{user_input}' 不在成语库中。")
            continue
        
        # 检查合法性
        legal_moves = state.get_legal_moves(graph)
        if found_id not in legal_moves:
            print(f"'{user_input}' 不是合法移动。")
            print(f"合法选项: {[dict_obj.get_text(id_) for id_ in legal_moves[:5]]}")
            continue
        
        # 用户移动
        state = state.make_move(found_id)
        print(f"你使用了: {user_input}")
        
        # AI 回应
        ai_move = solver.find_best_move(state)
        if ai_move is None:
            print("AI 无法接龙，你赢了！")
            break
        
        state = state.make_move(ai_move[0])
        print(f"AI 回应: {ai_move[1]}")


def main():
    """主函数"""
    print("成语接龙博弈求解器")
    print("="*50)
    print("选项:")
    print("1. 基础演示")
    print("2. Minimax 演示")
    print("3. 开局分析")
    print("4. 运行完整实验")
    print("5. 交互式游戏")
    print("0. 退出")
    
    while True:
        choice = input("\n请选择: ").strip()
        
        if choice == '1':
            demo_basic()
        elif choice == '2':
            demo_minimax()
        elif choice == '3':
            demo_analysis()
        elif choice == '4':
            run_full_experiment()
        elif choice == '5':
            interactive_game()
        elif choice == '0':
            print("退出程序。")
            break
        else:
            print("无效选项，请重新选择。")


if __name__ == "__main__":
    main()