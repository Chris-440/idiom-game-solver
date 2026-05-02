#!/usr/bin/env python3
"""
找到最大连通分量（节点数不超过1000），用于验证有环情况
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_data import IdiomDictionary
from src.idiom_graph import IdiomGraph


def main():
    idiom_file = "/Users/dzj/code/成语接龙/data/chinese-xinhua-master/data/idiom.json"
    
    # 加载所有成语
    with open(idiom_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dict_obj = IdiomDictionary(use_pinyin=True)
    for i, item in enumerate(data):
        word = item.get('word', '')
        pinyin = item.get('pinyin', '')
        if word and len(word) == 4:
            dict_obj.add_idiom(i, word, pinyin)
    
    print(f"加载成语总数: {len(dict_obj)}")
    
    # 构建图（启用音同匹配）
    print("构建成语图（音同匹配）...")
    graph = IdiomGraph(dict_obj, use_pinyin=True)
    print(f"总节点数: {graph.num_nodes}, 总边数: {graph.num_edges}")
    print(f"连通分量数: {graph.num_components}")
    
    # 按大小排序连通分量
    sorted_components = sorted(graph.components, key=len, reverse=True)
    
    print("\n前10大连通分量:")
    for i, comp in enumerate(sorted_components[:10]):
        print(f"  分量{i+1}: {len(comp)}个节点")

    # 从最大分量中提取1000个节点（使用BFS保持连通性）
    largest = sorted_components[0]
    
    if len(largest) <= 100:
        target_comp = largest
        print(f"\n最大分量正好{len(target_comp)}个节点")
    else:
        # 从最大分量中BFS扩展选取100个（保持连通性）
        start_node = next(iter(largest))
        
        # BFS扩展到100个节点（保持连通性）
        from collections import deque
        visited = {start_node}
        queue = deque([start_node])
        target_comp = set()
        
        while queue and len(target_comp) < 100:
            node = queue.popleft()
            target_comp.add(node)
            
            # 正向和反向邻居
            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited and neighbor in largest:
                    visited.add(neighbor)
                    queue.append(neighbor)
            
            for pred in graph.get_predecessors(node):
                if pred not in visited and pred in largest:
                    visited.add(pred)
                    queue.append(pred)
        
        print(f"\n最大分量超过1000，BFS扩展选取{len(target_comp)}个节点（保持连通）")
    
    # 保存选中的成语ID
    output_file = "/Users/dzj/code/成语接龙/results/selected_component.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存分量信息
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'component_size': len(target_comp),
            'component_ids': list(target_comp),
            'idioms': [dict_obj.get_text(id_) for id_ in target_comp],
        }, f, ensure_ascii=False, indent=2)
    
    print(f"已保存到: {output_file}")
    
    # 分析这个分量的内部边数
    internal_edges = 0
    for node in target_comp:
        for neighbor in graph.get_neighbors(node):
            if neighbor in target_comp:
                internal_edges += 1
    
    print(f"分量内部边数: {internal_edges}")
    if len(target_comp) > 1:
        print(f"分量密度: {internal_edges / (len(target_comp) * (len(target_comp) - 1)):.6f}")
    else:
        print(f"分量密度: N/A (只有1个节点)")


if __name__ == "__main__":
    main()
