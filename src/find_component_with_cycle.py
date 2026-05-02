#!/usr/bin/env python3
"""
找到包含环的子图（所有节点出度>0）
策略：找出所有强连通分量，只保留有环的SCC
"""

import json
import sys
import os
import random
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_data import IdiomDictionary


def find_sccs(adjacency, nodes):
    """Tarjan算法找强连通分量"""
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    
    def strongconnect(v):
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        
        for w in adjacency.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack.get(w, False):
                lowlinks[v] = min(lowlinks[v], index[w])
        
        if lowlinks[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)
    
    # 迭代版本避免递归栈溢出
    def strongconnect_iterative(start):
        call_stack = [(start, iter(adjacency.get(start, [])), None)]
        index[start] = index_counter[0]
        lowlinks[start] = index_counter[0]
        index_counter[0] += 1
        stack.append(start)
        on_stack[start] = True
        
        while call_stack:
            v, neighbors, child_result = call_stack[-1]
            
            try:
                w = next(neighbors)
                if w not in index:
                    index[w] = index_counter[0]
                    lowlinks[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    call_stack.append((w, iter(adjacency.get(w, [])), None))
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], index[w])
            except StopIteration:
                if lowlinks[v] == index[v]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)
                
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[v])
    
    for v in nodes:
        if v not in index:
            strongconnect_iterative(v)
    
    return sccs


def main():
    idiom_file = "/Users/dzj/code/成语接龙/data/chinese-xinhua-master/data/idiom.json"
    
    with open(idiom_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载成语
    dict_obj = IdiomDictionary(use_pinyin=True)
    for i, item in enumerate(data[:5000]):
        word = item.get('word', '')
        pinyin = item.get('pinyin', '')
        if word and len(word) == 4:
            dict_obj.add_idiom(i, word, pinyin)
    
    print(f"加载成语数: {len(dict_obj)}")
    
    # 构建邻接表
    adjacency = defaultdict(list)
    for idiom_id in dict_obj.get_all_ids():
        followers = dict_obj.get_followers(idiom_id, use_pinyin=True)
        adjacency[idiom_id] = followers
    
    print(f"总边数: {sum(len(v) for v in adjacency.values())}")
    
    # 找强连通分量
    print("\n查找强连通分量...")
    nodes = dict_obj.get_all_ids()
    sccs = find_sccs(adjacency, nodes)
    
    # 过滤出大小>1的SCC（有环）
    cyclic_sccs = [scc for scc in sccs if len(scc) > 1]
    cyclic_sccs.sort(key=len, reverse=True)
    
    print(f"强连通分量数: {len(sccs)}")
    print(f"有环的SCC数: {len(cyclic_sccs)}")
    print(f"最大有环SCC: {len(cyclic_sccs[0]) if cyclic_sccs else 0}个节点")
    
    # 选择最大的有环SCC
    if cyclic_sccs:
        largest_cyclic = set(cyclic_sccs[0])
        print(f"\n最大有环SCC: {len(largest_cyclic)}个节点")
        
        # 从最大SCC中随机选100个，但要保持连通性
        # 从随机节点开始BFS
        start = random.choice(list(largest_cyclic))
        target_nodes = set()
        visited = {start}
        queue = deque([start])
        
        while queue and len(target_nodes) < 100:
            node = queue.popleft()
            target_nodes.add(node)
            
            # 只在该SCC内扩展
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited and neighbor in largest_cyclic:
                    visited.add(neighbor)
                    queue.append(neighbor)
            
            for pred_node in largest_cyclic:
                if pred_node not in visited and node in adjacency.get(pred_node, []):
                    visited.add(pred_node)
                    queue.append(pred_node)
        
        print(f"BFS扩展选中的节点数: {len(target_nodes)}")
    else:
        print("错误: 没有找到环!")
        sys.exit(1)
    
    # 验证：每个节点在子图中出度>0
    sub_adjacency = defaultdict(list)
    for node in target_nodes:
        for neighbor in adjacency.get(node, []):
            if neighbor in target_nodes:
                sub_adjacency[node].append(neighbor)
    
    zero_out = [n for n in target_nodes if len(sub_adjacency[n]) == 0]
    print(f"子图中出度为0的节点数: {len(zero_out)}")
    
    # 迭代移除出度为0的节点直到稳定
    while zero_out:
        print(f"移除 {len(zero_out)} 个出度为0的节点...")
        target_nodes -= set(zero_out)
        
        sub_adjacency = defaultdict(list)
        for node in target_nodes:
            for neighbor in adjacency.get(node, []):
                if neighbor in target_nodes:
                    sub_adjacency[node].append(neighbor)
        
        zero_out = [n for n in target_nodes if len(sub_adjacency[n]) == 0]
    
    print(f"最终保留的节点数: {len(target_nodes)}")
    
    # 创建子图字典
    sub_dict = IdiomDictionary(use_pinyin=True)
    for idiom_id in target_nodes:
        if idiom_id in dict_obj.id_to_text:
            sub_dict.add_idiom(idiom_id, dict_obj.id_to_text[idiom_id], 
                             dict_obj.id_to_pinyin.get(idiom_id))
    
    print(f"\n最终子图:")
    print(f"  节点数: {len(sub_dict)}")
    print(f"  边数: {sum(len(v) for v in sub_adjacency.values())}")
    
    # 保存
    output_file = "/Users/dzj/code/成语接龙/results/selected_component.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'component_size': len(target_nodes),
            'component_ids': list(target_nodes),
            'idioms': [dict_obj.id_to_text[id_] for id_ in target_nodes if id_ in dict_obj.id_to_text],
            'has_cycle': True,
            'zero_out_degree': len(zero_out),
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存到: {output_file}")


if __name__ == "__main__":
    main()
