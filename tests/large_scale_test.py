#!/usr/bin/env python3
"""
从 chinese-xinhua 数据库加载成语数据
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_data import IdiomDictionary
from src.idiom_graph import IdiomGraph, GameState
from src.sg_solver import SGSolver
from src.minimax_solver import MinimaxSolver
from src.config import IDIOM_FILE, get_result_path


def load_real_idioms(filepath: str, limit: int = None) -> IdiomDictionary:
    """
    从 chinese-xinhua 数据库加载成语
    
    Args:
        filepath: idiom.json 文件路径
        limit: 加载数量限制（用于测试）
    
    Returns:
        IdiomDictionary 对象
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dict_obj = IdiomDictionary(use_pinyin=True)
    
    for i, item in enumerate(data):
        if limit and i >= limit:
            break
        
        word = item.get('word', '')
        pinyin = item.get('pinyin', '')
        
        if word and len(word) == 4:
            dict_obj.add_idiom(i, word, pinyin)
    
    return dict_obj


def test_scale(n_idioms: int, filepath: str):
    """
    测试不同规模的成语数据
    
    Args:
        n_idioms: 成语数量
        filepath: 数据文件路径
    """
    import time
    
    print(f"\n{'='*50}")
    print(f"大规模测试 (n={n_idioms})")
    print(f"{'='*50}")
    
    # 加载数据
    print("加载成语数据...")
    start = time.time()
    dict_obj = load_real_idioms(filepath, limit=n_idioms)
    load_time = time.time() - start
    print(f"加载时间: {load_time:.2f}s")
    print(f"成语数量: {len(dict_obj)}")
    
    # 构建图
    print("\n构建成语图...")
    start = time.time()
    graph = IdiomGraph(dict_obj)
    build_time = time.time() - start
    print(f"构建时间: {build_time:.2f}s")
    print(f"图统计: {graph.get_stats()}")
    
    # SG求解
    print("\nSG求解...")
    solver = SGSolver(graph, max_cache_size=10000000)  # 10M缓存
    
    start = time.time()
    initial_sg = solver.calculate_sg_initial()
    solve_time = time.time() - start
    
    print(f"求解时间: {solve_time:.2f}s")
    print(f"初始SG值: {initial_sg}")
    print(f"先手必胜: {initial_sg > 0}")
    print(f"缓存统计: {solver.get_cache_stats()}")
    
    # 开局分析（抽样）
    print("\n开局分析（抽样100个）...")
    sample_ids = list(dict_obj.get_all_ids())[:100]
    
    winning_count = 0
    losing_count = 0
    
    start = time.time()
    for idiom_id in sample_ids:
        state = GameState().make_move(idiom_id)
        sg = solver.calculate_sg(state)
        if sg == 0:
            winning_count += 1
        else:
            losing_count += 1
    analysis_time = time.time() - start
    
    print(f"分析时间: {analysis_time:.2f}s")
    print(f"抽样开局数: {len(sample_ids)}")
    print(f"必胜开局: {winning_count} ({winning_count/len(sample_ids)*100:.1f}%)")
    print(f"必败开局: {losing_count} ({losing_count/len(sample_ids)*100:.1f}%)")
    
    return {
        'n_idioms': n_idioms,
        'load_time': load_time,
        'build_time': build_time,
        'solve_time': solve_time,
        'initial_sg': initial_sg,
        'is_winning': initial_sg > 0,
        'cache_stats': solver.get_cache_stats(),
        'analysis_time': analysis_time,
        'winning_rate': winning_count / len(sample_ids) * 100,
    }


def main():
    """主函数"""
    filepath = IDIOM_FILE
    
    print("成语接龙大规模测试")
    print("="*50)
    
    # 测试不同规模
    results = []
    
    for n in [100, 500, 1000, 2000, 5000]:
        try:
            result = test_scale(n, filepath)
            results.append(result)
        except Exception as e:
            print(f"测试 n={n} 失败: {e}")
            break
    
    # 输出汇总
    print("\n" + "="*50)
    print("测试汇总")
    print("="*50)
    
    for r in results:
        print(f"n={r['n_idioms']}: 求解时间={r['solve_time']:.2f}s, "
              f"SG={r['initial_sg']}, 必胜={r['is_winning']}, "
              f"胜率={r['winning_rate']:.1f}%")
    
    # 保存结果
    import json
    with open(get_result_path('large_scale_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 results/large_scale_results.json")


if __name__ == "__main__":
    main()