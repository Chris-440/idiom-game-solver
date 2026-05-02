#!/usr/bin/env python3
"""
方法B: Q-Learning (边级Q表)
学习每条边 Q(u,v) 的价值
"""

import sys
import os
import random
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.idiom_graph import IdiomGraph


class QLearningPlayer:
    """基于Q-Learning的选手"""
    
    def __init__(self, graph, q_table=None, name="QLearning"):
        self.graph = graph
        self.q_table = q_table or defaultdict(dict)
        self.name = name
        
        # 初始化Q表
        if not q_table:
            for u in graph.dictionary.get_all_ids():
                for v in graph.get_neighbors(u):
                    self.q_table[u][v] = random.uniform(-0.1, 0.1)
    
    def select_move(self, current_idiom, used_set):
        """ε-greedy选择"""
        neighbors = self.graph.get_neighbors(current_idiom)
        valid_moves = [n for n in neighbors if n not in used_set]
        
        if not valid_moves:
            return None
        
        # 选择Q值最大的动作
        best_move = valid_moves[0]
        best_q = -float('inf')
        
        for v in valid_moves:
            q = self.q_table.get(current_idiom, {}).get(v, 0.0)
            if q > best_q:
                best_q = q
                best_move = v
        
        return best_move
    
    def get_name(self):
        return self.name
    
    @classmethod
    def train(cls, graph: IdiomGraph, episodes=30000, lr=0.02, gamma=0.98, epsilon_start=0.3):
        """训练Q-Learning选手"""
        player = cls(graph)
        q_table = player.q_table
        
        print(f"训练 {player.name}: {episodes} 局...")
        
        wins = {0: 0, 1: 0}
        total_steps = 0
        epsilon = epsilon_start
        
        all_nodes = list(graph.dictionary.get_all_ids())
        
        for ep in range(episodes):
            # 模拟一局
            current = random.choice(all_nodes)
            used = {current}
            history = []
            p = 0
            
            for _ in range(500):
                neighbors = graph.get_neighbors(current)
                valid = [n for n in neighbors if n not in used]
                
                if not valid:
                    winner = 1 - p
                    break
                
                # ε-greedy
                if random.random() < epsilon:
                    move = random.choice(valid)
                else:
                    best_q = -float('inf')
                    best_move = valid[0]
                    for v in valid:
                        q = q_table.get(current, {}).get(v, 0.0)
                        if q > best_q:
                            best_q = q
                            best_move = v
                    move = best_move
                
                history.append((p, current, move))
                used.add(move)
                current = move
                p = 1 - p
            
            total_steps += len(history)
            
            # 更新Q值
            n = len(history)
            for i in range(n - 1, -1, -1):
                player_id, from_u, to_v = history[i]
                
                if i == n - 1:
                    reward = 1.0 if player_id == winner else -1.0
                else:
                    reward = 0.5 if player_id == winner else -0.5
                
                old_q = q_table.get(from_u, {}).get(to_v, 0.0)
                
                if i < n - 1:
                    next_from = history[i+1][1]
                    next_vals = list(q_table.get(next_from, {}).values()) or [0.0]
                    next_max = max(next_vals)
                else:
                    next_max = 0.0
                
                new_q = old_q + lr * (reward + gamma * (-next_max) - old_q)
                q_table[from_u][to_v] = new_q
            
            wins[winner if 'winner' in dir() else 0] += 1
            epsilon = epsilon_start * (1 - ep / episodes) ** 2 + 0.02
            
            if (ep + 1) % 5000 == 0:
                print(f"  {ep+1}/{episodes}: P0={wins[0]}, P1={wins[1]}, ε={epsilon:.3f}")
        
        print(f"训练完成: 总步数={total_steps}")
        return player
    
    def get_name(self):
        return self.name
