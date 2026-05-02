#!/usr/bin/env python3
"""
基于价值迭代 (Value Iteration) 的大规模成语接龙求解器
这是处理 3w+ 级别成语、有环图的最优方法
相比 SG 算法，它不记录 used_set，而是学习每个成语的"固有胜率/价值"
"""

import sys
import os
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_graph import IdiomGraph


class ValueIterationSolver:
    def __init__(self, graph: IdiomGraph, iterations=50, gamma=0.99):
        self.graph = graph
        self.iterations = iterations
        self.gamma = gamma  # 折扣因子，接近 1
        
        # V[idiom_id] = value (期望价值)
        self.values = {}
        
        # 缓存最佳移动
        self.best_moves = {}

    def solve(self):
        """运行价值迭代算法"""
        print(f"开始价值迭代，共 {self.iterations} 轮...")
        
        all_nodes = list(self.graph.dictionary.get_all_ids())
        
        # 初始化价值为 0
        for node in all_nodes:
            self.values[node] = 0.0
        
        # 构建邻接表以加速访问
        neighbors_map = {}
        for node in all_nodes:
            neighbors_map[node] = list(self.graph.get_neighbors(node))
        
        # 迭代更新
        for i in range(self.iterations):
            new_values = {}
            delta = 0.0
            
            for u in all_nodes:
                neighbors = neighbors_map[u]
                
                if not neighbors:
                    # 无路可走，必败态，价值 -1
                    new_values[u] = -1.0
                    continue
                
                # 价值公式: V(u) = max( 1 (如果 v 是死路) 或 -gamma * V(v) )
                # 解释：
                # 如果我走到 v，且 v 无路可走，我直接赢，价值 +1
                # 如果我走到 v，且 v 有路可走，对手的价值是 V(v)
                # 因为是零和博弈，对手的价值就是我的代价，所以我是 -V(v)
                # 乘以 gamma 是因为未来的价值比现在的略低（鼓励快速获胜）
                
                best_val = -float('inf')
                best_move = None
                
                for v in neighbors:
                    # 计算走到 v 的价值
                    if not neighbors_map[v]:
                        val = 1.0  # 对手无路可走，我赢
                    else:
                        val = -self.gamma * self.values[v]
                    
                    if val > best_val:
                        best_val = val
                        best_move = v
                
                new_values[u] = best_val
                self.best_moves[u] = best_move
                
                delta = max(delta, abs(new_values[u] - self.values[u]))
            
            self.values = new_values
            
            if (i + 1) % 10 == 0:
                print(f"  轮次 {i + 1}/{self.iterations}, 最大变化: {delta:.4f}")
            
            # 如果收敛，提前退出
            if delta < 1e-4:
                print(f"  在第 {i + 1} 轮收敛")
                break
        
        print("求解完成。")

    def get_best_move(self, idiom_id):
        """获取最佳接龙成语"""
        if idiom_id not in self.best_moves:
            return None, None
        
        best_id = self.best_moves[idiom_id]
        if best_id is None:
            return None, None
            
        return best_id, self.graph.dictionary.get_text(best_id)

    def analyze_all_idioms(self):
        """分析所有成语的胜负态"""
        results = {}
        for idiom_id in self.graph.dictionary.get_all_ids():
            val = self.values.get(idiom_id, 0.0)
            
            # 阈值判定：
            # Value > 0.1 为必胜（因为存在折扣因子，纯环可能是 0 或小正值）
            # Value <= 0.1 为必败或复杂态
            state = 'WIN' if val > 0.05 else 'LOSE'
            
            best_id, best_text = self.get_best_move(idiom_id)
            
            results[idiom_id] = {
                'text': self.graph.dictionary.get_text(idiom_id),
                'state': state,
                'value': val,
                'best_next_id': best_id,
                'best_next_text': best_text
            }
        return results
