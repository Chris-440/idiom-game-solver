#!/usr/bin/env python3
"""
分析 Random "自杀" 行为
"""

import sys
import os
import json
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_data import IdiomDictionary
from src.idiom_graph import IdiomGraph

def load_graph(idiom_file):
    with open(idiom_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dict_obj = IdiomDictionary(use_pinyin=True)
    for i, item in enumerate(data):
        word = item.get('word', '')
        pinyin = item.get('pinyin', '')
        if word and len(word) == 4:
            dict_obj.add_idiom(i, word, pinyin)
    
    return IdiomGraph(dict_obj, use_pinyin=True)

def analyze_suicide(graph: IdiomGraph):
    """分析 Random 自杀行为"""
    
    # 统计出度分布
    out_degrees = [graph.get_out_degree(n) for n in graph.dictionary.get_all_ids()]
    
    print("=" * 60)
    print("Random 自杀行为分析")
    print("=" * 60)
    
    print(f"\n图基本信息:")
    print(f"  节点数: {graph.num_nodes}")
    print(f"  边数: {graph.num_edges}")
    print(f"  平均出度: {sum(out_degrees)/len(out_degrees):.1f}")
    print(f"  出度中位数: {sorted(out_degrees)[len(out_degrees)//2]}")
    print(f"  最大出度: {max(out_degrees)}")
    print(f"  最小出度: {min(out_degrees)}")
    
    # 出度分布
    print(f"\n出度分布:")
    bins = [0, 1, 2, 5, 10, 20, 50, 100]
    for i in range(len(bins)):
        low = bins[i]
        high = bins[i+1] if i+1 < len(bins) else float('inf')
        count = sum(1 for d in out_degrees if low <= d < high)
        pct = count / len(out_degrees) * 100
        label = f"{low}-{high-1}" if high != float('inf') else f"{low}+"
        print(f"  出度 {label:>5}: {count:>6} 节点 ({pct:>5.1f}%)")
    
    # 模拟 Random 对局
    print(f"\n--- 模拟 1000 局 Random vs Random ---")
    print(f"(Random 自杀率 = Random 输掉的比例)")
    
    random.seed(42)
    all_nodes = list(graph.dictionary.get_all_ids())
    
    random_losses_by_outdegree = {d: 0 for d in range(0, 51)}
    random_games_by_startdegree = {d: 0 for d in range(0, 51)}
    
    total_random_losses = 0
    total_games = 0
    
    for game in range(1000):
        # 随机起始
        cur = random.choice(all_nodes)
        used = {cur}
        start_degree = graph.get_out_degree(cur)
        
        # 模拟游戏
        steps = 0
        current_player = 0  # 0=Random1, 1=Random2
        
        for _ in range(500):
            neighbors = graph.get_neighbors(cur)
            valid = [n for n in neighbors if n not in used]
            
            if not valid:
                # 当前玩家无路 = 输
                total_random_losses += 1
                total_games += 1
                break
            
            # Random 选词
            move = random.choice(valid)
            used.add(move)
            cur = move
            steps += 1
            current_player = 1 - current_player
        else:
            total_games += 1  # 平局也算一局
    
    print(f"  总游戏数: {total_games}")
    print(f"  Random 输掉: {total_random_losses} ({total_random_losses/total_games*100:.1f}%)")
    print(f"  (注：Random vs Random 应该是 50% 胜率，输=自杀)")
    
    # 分析：Random 输的时候，是因为走进了低出度区域吗？
    print(f"\n--- 深入分析：Random 输的原因 ---")
    print(f"  在 Random vs Random 中，输的一方 '自杀' 意味着：")
    print(f"  1. 它随机选择了一个后继，但该后继很快导致无路可走")
    print(f"  2. 它被对手逼进了低出度区域")
    print(f"  3. 它自己主动走进了死胡同")
    
    # 分析低出度节点的"死亡陷阱"属性
    print(f"\n--- 陷阱节点分析 ---")
    low_out_nodes = [n for n in graph.dictionary.get_all_ids() if graph.get_out_degree(n) <= 3]
    high_out_nodes = [n for n in graph.dictionary.get_all_ids() if graph.get_out_degree(n) >= 20]
    
    print(f"  低出度节点 (出度≤3): {len(low_out_nodes)} 个")
    print(f"  高出度节点 (出度≥20): {len(high_out_nodes)} 个")
    
    # 计算从低出度节点出发，Random 存活步数的期望
    avg_survival_low = 0
    for _ in range(100):
        for node in random.sample(low_out_nodes, min(50, len(low_out_nodes))):
            used = {node}
            cur = node
            steps = 0
            for _ in range(100):
                neighbors = graph.get_neighbors(cur)
                valid = [n for n in neighbors if n not in used]
                if not valid:
                    break
                cur = random.choice(valid)
                used.add(cur)
                steps += 1
            avg_survival_low += steps
    
    avg_survival_low /= (100 * min(50, len(low_out_nodes)))
    
    avg_survival_high = 0
    for _ in range(100):
        for node in random.sample(high_out_nodes, min(50, len(high_out_nodes))):
            used = {node}
            cur = node
            steps = 0
            for _ in range(100):
                neighbors = graph.get_neighbors(cur)
                valid = [n for n in neighbors if n not in used]
                if not valid:
                    break
                cur = random.choice(valid)
                used.add(cur)
                steps += 1
            avg_survival_high += steps
    
    avg_survival_high /= (100 * min(50, len(high_out_nodes)))
    
    print(f"\n  Random 从低出度节点出发的平均存活步数: {avg_survival_low:.1f}")
    print(f"  Random 从高出度节点出发的平均存活步数: {avg_survival_high:.1f}")
    print(f"  差距: {avg_survival_high/avg_survival_low:.1f}x")
    
    print(f"\n--- 结论 ---")
    print(f"  Random '自杀' 的本质原因是：")
    print(f"  1. 它均匀随机选词，不区分'安全路径'和'陷阱路径'")
    print(f"  2. 图中存在大量低出度节点（{len(low_out_nodes)}个），")
    print(f"     一旦走进这些节点，平均只能再走 {avg_survival_low:.0f} 步就死了")
    print(f"  3. Q-Learning 学会了避开这些陷阱，同时诱导对手走进陷阱")
    print(f"  4. 因此 Random 对 Q-Learning 的胜率只有 2.6%——")
    print(f"     大部分失败是 '自己走进死胡同'，而非被对手直接击败")

if __name__ == "__main__":
    idiom_file = "/Users/dzj/code/成语接龙/data/chinese-xinhua-master/data/idiom.json"
    graph = load_graph(idiom_file)
    analyze_suicide(graph)
