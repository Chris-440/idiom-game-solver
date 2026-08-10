#!/usr/bin/env python3
"""
方法A: Value Iteration (不考虑 used_set)
学习每条边 (u→v) 的长期价值 V(v)，选择价值最高的后继
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.idiom_graph import IdiomGraph


class ValueIterationPlayer:
    """基于价值迭代的选手"""
    
    def __init__(self, graph: IdiomGraph, iterations=100, gamma=0.99):
        self.graph = graph
        self.gamma = gamma
        self.values = {}
        self.name = "ValueIteration"
        
        # 初始化价值
        for node in graph.dictionary.get_all_ids():
            self.values[node] = 0.0
        
        # 构建邻接表
        self.neighbors = {}
        for node in graph.dictionary.get_all_ids():
            self.neighbors[node] = list(graph.get_neighbors(node))
        
        # 运行价值迭代
        self._solve(iterations)
    
    def _solve(self, iterations):
        """运行价值迭代"""
        for _ in range(iterations):
            new_values = {}
            for u in self.graph.dictionary.get_all_ids():
                neighbors = self.neighbors[u]
                if not neighbors:
                    new_values[u] = -1.0
                    continue
                
                best_val = -float('inf')
                for v in neighbors:
                    if not self.neighbors[v]:
                        val = 1.0
                    else:
                        val = -self.gamma * self.values[v]
                    best_val = max(best_val, val)
                
                new_values[u] = best_val
            
            self.values = new_values
    
    def select_move(self, current_idiom, used_set):
        """选择最佳移动"""
        valid_moves = [n for n in self.neighbors.get(current_idiom, []) if n not in used_set]
        
        if not valid_moves:
            return None
        
        # 选择价值最高的后继
        best_move = None
        best_val = -float('inf')
        
        for v in valid_moves:
            val = self.values.get(v, 0.0)
            if val > best_val:
                best_val = val
                best_move = v
        
        return best_move
    
    def get_name(self):
        return self.name
