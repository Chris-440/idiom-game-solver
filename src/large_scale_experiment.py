#!/usr/bin/env python3
"""
大规模实验脚本 - 全量数据 (3w+)
使用价值迭代 (Value Iteration) 算法
"""

import json
import time
import sys
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_data import IdiomDictionary
from src.idiom_graph import IdiomGraph, GameState
from src.q_solver import ValueIterationSolver
from src.selfplay_solver import SelfPlaySolver


def load_all_idioms(filepath: str) -> IdiomDictionary:
    """加载所有成语"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dict_obj = IdiomDictionary(use_pinyin=True)
    count = 0
    for i, item in enumerate(data):
        word = item.get('word', '')
        pinyin = item.get('pinyin', '')
        if word and len(word) == 4:
            dict_obj.add_idiom(i, word, pinyin)
            count += 1
    
    return dict_obj


def prune_zero_out_degree_nodes(dict_obj: IdiomDictionary) -> IdiomDictionary:
    """
    迭代移除出度为0的节点，直到图中所有节点出度>0
    
    这相当于找到图的最大子图，其中每个节点都有后继。
    出度为0的成语在博弈中等同于"一说就赢"，这是不合理的。
    """
    # 初始化：所有节点都保留
    active_nodes = set(dict_obj.get_all_ids())
    
    # 构建初始邻接表（只考虑active_nodes内的节点）
    def build_adj(nodes):
        adj = {}
        for nid in nodes:
            followers = dict_obj.get_followers(nid, use_pinyin=True)
            # 只保留在active_nodes中的后继
            valid = [f for f in followers if f in nodes]
            adj[nid] = valid
        return adj
    
    iteration = 0
    total_removed = 0
    
    while True:
        iteration += 1
        adj = build_adj(active_nodes)
        
        # 找出出度为0的节点
        zero_out = [nid for nid, nbrs in adj.items() if len(nbrs) == 0]
        
        if not zero_out:
            break
        
        # 移除这些节点
        active_nodes -= set(zero_out)
        total_removed += len(zero_out)
        print(f"  第{iteration}轮: 移除 {len(zero_out)} 个出度为0的节点 (剩余: {len(active_nodes)})")
        
        if len(active_nodes) == 0:
            print("  警告: 所有节点都被移除了!")
            break
    
    print(f"  总计移除: {total_removed} 个节点")
    print(f"  最终保留: {len(active_nodes)} 个节点")
    
    # 用保留的节点创建新字典
    pruned_dict = IdiomDictionary(use_pinyin=True)
    for nid in active_nodes:
        text = dict_obj.get_text(nid)
        pinyin = dict_obj.get_pinyin(nid)
        pruned_dict.add_idiom(nid, text, pinyin)
    
    return pruned_dict


def analyze_graph_density(graph: IdiomGraph) -> Dict:
    """分析图密度"""
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    max_edges = num_nodes * (num_nodes - 1)
    density = num_edges / max_edges if max_edges > 0 else 0
    
    out_degrees = [graph.get_out_degree(n) for n in graph.dictionary.get_all_ids()]
    in_degrees = [graph.get_in_degree(n) for n in graph.dictionary.get_all_ids()]
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'sparsity': 1 - density,
        'avg_out_degree': sum(out_degrees) / len(out_degrees),
        'max_out_degree': max(out_degrees),
        'zero_out_degree_count': sum(1 for d in out_degrees if d == 0),
        'num_components': graph.num_components,
        'max_component_size': graph.max_component_size,
    }


def build_web_data(graph: IdiomGraph, solver: ValueIterationSolver, 
                   graph_analysis: Dict, results: Dict) -> Dict:
    """构建 Web 数据（节点全量保留，边采样用于可视化）"""
    nodes = []
    
    print("  正在构建节点数据...")
    for idiom_id in graph.dictionary.get_all_ids():
        text = graph.dictionary.get_text(idiom_id)
        res = results.get(idiom_id, {})
        
        nodes.append({
            'id': idiom_id,
            'text': text,
            'out_degree': graph.get_out_degree(idiom_id),
            'in_degree': graph.get_in_degree(idiom_id),
            'sg_value': res.get('value', 0),
            'is_winning': res.get('state') == 'WIN',
            'is_losing': res.get('state') == 'LOSE',
        })
    
    # 边采样：保留全量统计，但可视化只输出部分边
    print("  正在采样边数据...")
    all_edges_count = 0
    sampled_edges = []
    max_edges = 50000  # 最多5万条边用于可视化
    
    # 策略：优先保留最优对答的边，然后随机采样
    optimal_from_ids = set()
    for idiom_id, res in results.items():
        if res.get('best_next_id'):
            optimal_from_ids.add(idiom_id)
    
    for from_id in graph.dictionary.get_all_ids():
        neighbors = list(graph.get_neighbors(from_id))
        for to_id in neighbors:
            all_edges_count += 1
            
            # 如果是必胜态且是最优对答路径上的，优先保留
            res = results.get(from_id, {})
            if res.get('state') == 'WIN' and res.get('best_next_id') == to_id:
                sampled_edges.append({'from': from_id, 'to': to_id})
            elif len(sampled_edges) < max_edges:
                # 随机采样
                if random.random() < 0.03:  # 约3%的采样率
                    sampled_edges.append({'from': from_id, 'to': to_id})
    
    print(f"  总边数: {all_edges_count}, 采样后: {len(sampled_edges)}")
    
    # 最优对答图：保留全部
    print("  正在构建最优对答数据...")
    optimal_edges = []
    for idiom_id, res in results.items():
        if res.get('best_next_id'):
            optimal_edges.append({
                'from': idiom_id,
                'to': res['best_next_id'],
                'from_text': res['text'],
                'to_text': res['best_next_text'],
                'is_winning': res['state'] == 'WIN',
                'is_winning_to': results.get(res['best_next_id'], {}).get('state') == 'WIN',
            })
    
    return {
        'graph': {'nodes': nodes, 'edges': sampled_edges},
        'optimal_graph': {'edges': optimal_edges},
        'analysis': {
            'graph_analysis': graph_analysis,
        },
        'metadata': {
            'total_idioms': len(nodes),
            'total_edges': all_edges_count,  # 显示全量统计
            'sampled_edges': len(sampled_edges),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    }


def run_experiment(idiom_file: str, output_dir: str = 'results'):
    """运行全量实验（价值迭代 + 出度剪枝分析）"""
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("全量成语实验 (价值迭代 + 出度剪枝分析)")
    print("="*60)

    # 1. 加载数据
    print("\n[1/5] 加载所有成语数据...")
    start_time = time.time()
    dict_obj = load_all_idioms(idiom_file)
    print(f"  加载完成: {len(dict_obj)} 个成语, 耗时 {time.time() - start_time:.2f}s")

    # 2. 构建初始图
    print("\n[2/5] 构建成语图（音同匹配）...")
    start_time = time.time()
    graph = IdiomGraph(dict_obj, use_pinyin=True)
    print(f"  构建完成: {graph.num_nodes} 节点, {graph.num_edges} 边, 耗时 {time.time() - start_time:.2f}s")
    
    # 2.5. 迭代剪枝：移除出度为0的节点（分析用）
    print("\n[2.5/5] 迭代移除出度为0的节点（分析用）...")
    start_time = time.time()
    pruned_dict = prune_zero_out_degree_nodes(dict_obj)
    prune_time = time.time() - start_time
    pruned_graph = IdiomGraph(pruned_dict, use_pinyin=True)
    print(f"  剪枝后图: {pruned_graph.num_nodes} 节点, {pruned_graph.num_edges} 边, 耗时 {prune_time:.2f}s")
    
    # 在剪枝后的图上运行价值迭代
    print("\n[2.6/5] 在剪枝后图上运行价值迭代...")
    pruned_solver = ValueIterationSolver(pruned_graph, iterations=100)
    pruned_solver.solve()
    pruned_results = pruned_solver.analyze_all_idioms()
    
    pruned_win = sum(1 for r in pruned_results.values() if r['state'] == 'WIN')
    pruned_lose = sum(1 for r in pruned_results.values() if r['state'] == 'LOSE')
    print(f"  剪枝后图 - 必胜态: {pruned_win}, 必败态: {pruned_lose}")

    # 3. 分析密度（用原始图）
    print("\n[3/5] 分析图密度...")
    graph_analysis = analyze_graph_density(graph)
    print(f"  密度: {graph_analysis['density']:.6f}")
    print(f"  连通分量数: {graph_analysis['num_components']}")
    print(f"  最大连通分量: {graph_analysis['max_component_size']}")

    # 4. Self-play Q-Learning 求解（跟踪 used_set）
    print("\n[4/5] 运行 Self-play Q-Learning (大规模训练)...")
    start_time = time.time()
    solver = SelfPlaySolver(graph, lr=0.02, gamma=0.98, epsilon=0.3)
    q_stats = solver.train(episodes=30000, verbose=True)
    results = solver.analyze_all_idioms()
    solve_time = time.time() - start_time
    print(f"  求解耗时: {solve_time:.2f}s")

    # 统计
    win_count = sum(1 for r in results.values() if r['state'] == 'WIN')
    lose_count = sum(1 for r in results.values() if r['state'] == 'LOSE')
    print(f"  必胜态: {win_count} ({win_count/len(results)*100:.2f}%)")
    print(f"  必败态: {lose_count} ({lose_count/len(results)*100:.2f}%)")
    print(f"  出度为0的节点（一说就赢）: {sum(1 for n in graph.dictionary.get_all_ids() if graph.get_out_degree(n) == 0)}")

    # 5. 构建并保存数据
    print("\n构建 Web 可视化数据...")
    web_data = build_web_data(graph, solver, graph_analysis, results)
    
    web_output_path = os.path.join(output_dir, 'web_visualization_data.json')
    print(f"保存 Web 数据到: {web_output_path}")
    with open(web_output_path, 'w', encoding='utf-8') as f:
        json.dump(web_data, f, ensure_ascii=False, indent=2)
    
    report = {
        'experiment_config': {
            'idiom_count': len(dict_obj),
            'algorithm': 'Self-play Q-Learning (跟踪 used_set)',
            'episodes': 3000,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'graph_analysis': graph_analysis,
        'results_summary': {
            'win_count': win_count,
            'lose_count': lose_count,
            'solve_time_seconds': round(solve_time, 3),
        },
    }
    
    report_path = os.path.join(output_dir, 'large_scale_experiment_report.json')
    print(f"保存实验报告到: {report_path}")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n实验全部完成!")


def main():
    idiom_file = "/Users/dzj/code/成语接龙/data/chinese-xinhua-master/data/idiom.json"
    
    if not os.path.exists(idiom_file):
        print(f"错误: 找不到成语数据文件 {idiom_file}")
        sys.exit(1)
    
    run_experiment(idiom_file, output_dir='results')


if __name__ == "__main__":
    main()
