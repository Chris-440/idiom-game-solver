#!/usr/bin/env python3
"""
方法C: MCTS (Monte Carlo Tree Search)
在线搜索，不预先训练
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.idiom_graph import IdiomGraph


class MCTSPlayer:
    """基于MCTS的选手"""
    
    def __init__(self, graph: IdiomGraph, simulations=1000, exploration=1.414):
        self.graph = graph
        self.simulations = simulations
        self.exploration = exploration
        self.name = "MCTS"
        
        # 构建邻接表
        self.neighbors = {}
        for node in graph.dictionary.get_all_ids():
            self.neighbors[node] = list(graph.get_neighbors(node))
    
    def select_move(self, current_idiom, used_set):
        """使用MCTS选择最佳移动"""
        valid_moves = [n for n in self.neighbors.get(current_idiom, []) if n not in used_set]
        
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # 对每个合法移动运行模拟
        move_stats = {m: 0 for m in valid_moves}
        
        for _ in range(self.simulations):
            # 选择一个移动开始模拟
            move = random.choice(valid_moves)
            
            # 模拟完整游戏
            result = self._simulate(current_idiom, move, set(used_set))
            
            if result == 1:  # 赢了
                move_stats[move] += 1
        
        # 选择胜率最高的移动
        best_move = max(valid_moves, key=lambda m: move_stats[m])
        return best_move
    
    def _simulate(self, start_from, first_move, used_set):
        """
        从当前状态模拟完整游戏
        返回: 1=先手赢, 0=后手赢
        """
        used = set(used_set)
        used.add(start_from)
        used.add(first_move)
        
        current = first_move
        player = 1  # 0=原始玩家, 1=对手
        
        for _ in range(500):  # 最大500步
            neighbors = self.neighbors.get(current, [])
            valid = [n for n in neighbors if n not in used]
            
            if not valid:
                # 当前玩家无路可走，对方赢
                return 1 - player
            
            # 随机选择（快速模拟）
            move = random.choice(valid)
            used.add(move)
            current = move
            player = 1 - player
        
        return 0  # 平局视为后手赢
    
    def get_name(self):
        return self.name
