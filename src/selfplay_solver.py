#!/usr/bin/env python3
"""
基于 Q-Learning + Self-play 的成语接龙求解器
关键：跟踪已使用的成语集合，处理"不可重复"规则
"""

import sys
import os
import random
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_graph import IdiomGraph


class SelfPlaySolver:
    """
    使用自对抗训练 + Q-Learning 求解成语接龙
    
    核心思想：
    - 学习每条边 (u→v) 的 Q 值，表示"从 u 走到 v"的期望胜率
    - 对局中跟踪 used_set，确保不重复
    - 使用 ε-greedy 探索策略
    """
    
    def __init__(self, graph: IdiomGraph, lr=0.05, gamma=0.95, epsilon=0.3):
        self.graph = graph
        self.lr = lr        # 学习率
        self.gamma = gamma  # 折扣因子（接近1，鼓励长期收益）
        self.epsilon = epsilon  # 探索率
        
        # Q表：Q[u][v] = 从u走到v的价值
        self.q_table = defaultdict(dict)
        
        # 初始化Q值（小随机值打破对称）
        for u in graph.dictionary.get_all_ids():
            for v in graph.get_neighbors(u):
                self.q_table[u][v] = random.uniform(-0.1, 0.1)
    
    def get_valid_moves(self, current_id, used_set):
        """获取合法移动（未使用过的后继）"""
        neighbors = self.graph.get_neighbors(current_id)
        return [n for n in neighbors if n not in used_set]
    
    def choose_action(self, current_id, used_set, training=True):
        """ε-greedy 策略选择动作"""
        valid_moves = self.get_valid_moves(current_id, used_set)
        
        if not valid_moves:
            return None  # 无路可走
        
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # 选择 Q 值最大的动作
        best_move = None
        best_q = -float('inf')
        
        for move in valid_moves:
            q = self.q_table[current_id].get(move, 0.0)
            if q > best_q:
                best_q = q
                best_move = move
        
        return best_move
    
    def play_episode(self, max_steps=500):
        """
        模拟一局游戏（self-play）
        返回: (game_history, winner)
        game_history = [(player_id, from_id, to_id), ...]
        """
        # 随机选择起始成语
        all_nodes = list(self.graph.dictionary.get_all_ids())
        current_id = random.choice(all_nodes)
        used_set = {current_id}
        
        history = []
        current_player = 0  # 0 或 1
        
        for step in range(max_steps):
            # 选择动作
            next_id = self.choose_action(current_id, used_set, training=True)
            
            if next_id is None:
                # 当前玩家无路可走，对手获胜
                winner = 1 - current_player
                return history, winner
            
            # 记录历史
            history.append((current_player, current_id, next_id))
            used_set.add(next_id)
            current_id = next_id
            current_player = 1 - current_player
        
        # 达到最大步数，视为平局（给双方-0.5）
        return history, -1  # -1 表示平局
    
    def update_q(self, history, winner):
        """
        根据游戏结果更新Q值
        
        更新策略：
        - 如果当前玩家赢，他的移动应该增加Q值
        - 如果对手赢，他的移动应该减少Q值
        - 使用逆向传播：从游戏结局向前更新
        
        Returns: 更新次数
        """
        if winner == -1:  # 平局
            return 0
        
        updates = 0
        
        # 逆向遍历历史记录
        n = len(history)
        for i in range(n - 1, -1, -1):
            player, from_id, to_id = history[i]
            
            # 计算这个移动的奖励
            if i == n - 1:
                if winner == player:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                if winner == player:
                    reward = 0.5
                else:
                    reward = -0.5
            
            old_q = self.q_table[from_id].get(to_id, 0.0)
            
            if i < n - 1:
                next_from = history[i + 1][1]
                next_vals = list(self.q_table[next_from].values()) if next_from in self.q_table else [0.0]
                next_max_q = max(next_vals) if next_vals else 0.0
            else:
                next_max_q = 0.0
            
            new_q = old_q + self.lr * (reward + self.gamma * (-next_max_q) - old_q)
            self.q_table[from_id][to_id] = new_q
            updates += 1
        
        return updates
    
    def train(self, episodes=500, verbose=True):
        """训练模型"""
        print(f"开始训练: {episodes} 局自对抗游戏...")
        
        wins = {0: 0, 1: 0}
        draws = 0
        total_q_updates = 0
        total_steps = 0
        
        # 记录每10%的Q值统计
        q_stats = []
        
        for ep in range(episodes):
            history, winner = self.play_episode()
            total_steps += len(history)
            n_updates = self.update_q(history, winner)
            total_q_updates += n_updates
            
            if winner == -1:
                draws += 1
            else:
                wins[winner] += 1
            
            # 逐渐降低探索率（更平滑的衰减）
            self.epsilon = 0.3 * (1 - ep / episodes) ** 2 + 0.02  # 最终保留2%探索
            
            if verbose and (ep + 1) % max(1, episodes // 20) == 0:
                # 计算Q值统计
                all_q = []
                for u in self.q_table:
                    for v in self.q_table[u]:
                        all_q.append(self.q_table[u][v])
                
                if all_q:
                    q_mean = sum(all_q) / len(all_q)
                    q_std = (sum((q - q_mean)**2 for q in all_q) / len(all_q)) ** 0.5
                    q_max = max(all_q)
                    q_min = min(all_q)
                    q_stats.append((ep + 1, q_mean, q_std, q_max, q_min))
                
                total = ep + 1
                print(f"  {total}/{episodes} ({total/episodes:.0%}): "
                      f"P0胜率={wins[0]/total:.1%}, P1胜率={wins[1]/total:.1%}, "
                      f"ε={self.epsilon:.3f}, "
                      f"avg_len={total_steps/total:.0f}步")
        
        print(f"\n训练完成: P0:{wins[0]}胜, P1:{wins[1]}胜, 平局:{draws}")
        print(f"总步数: {total_steps}, 总Q更新: {total_q_updates}")
        print(f"平均每个节点被访问: {total_steps / len(self.graph.dictionary.get_all_ids()):.1f}次")
        
        if q_stats:
            print(f"\nQ值收敛分析:")
            print(f"  {'轮次':>6} | {'均值':>8} | {'标准差':>8} | {'最大':>8} | {'最小':>8}")
            print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            for ep, mean, std, maxq, minq in q_stats:
                print(f"  {ep:6d} | {mean:8.4f} | {std:8.4f} | {maxq:8.4f} | {minq:8.4f}")
        
        return q_stats
    
    def analyze_all_idioms(self):
        """分析所有成语的必胜/必败态"""
        results = {}
        
        for idiom_id in self.graph.dictionary.get_all_ids():
            neighbors = self.graph.get_neighbors(idiom_id)
            
            if not neighbors:
                results[idiom_id] = {
                    'text': self.graph.dictionary.get_text(idiom_id),
                    'state': 'WIN',  # 出度为0，一说就赢
                    'value': 1.0,
                    'best_next_id': None,
                    'best_next_text': None,
                }
                continue
            
            # 找出最优移动
            best_id = None
            best_q = -float('inf')
            
            for n in neighbors:
                q = self.q_table[idiom_id].get(n, 0.0)
                if q > best_q:
                    best_q = q
                    best_id = n
            
            # 归一化：Q值范围约在[-1, 1]
            # 如果最佳移动Q > 0.1，认为是必胜态
            # 如果所有移动Q < -0.1，认为是必败态
            # 否则是复杂态
            if best_q > 0.1:
                state = 'WIN'
            elif best_q < -0.1:
                state = 'LOSE'
            else:
                # 根据Q值正负决定
                state = 'WIN' if best_q > 0 else 'LOSE'
            
            results[idiom_id] = {
                'text': self.graph.dictionary.get_text(idiom_id),
                'state': state,
                'value': best_q,
                'best_next_id': best_id,
                'best_next_text': self.graph.dictionary.get_text(best_id) if best_id else None,
            }
        
        return results
    
    def get_best_move(self, idiom_id, used_set=None):
        """获取最佳移动（考虑已使用集合）"""
        if used_set is None:
            used_set = set()
        
        valid_moves = self.get_valid_moves(idiom_id, used_set)
        if not valid_moves:
            return None, None
        
        best_id = None
        best_q = -float('inf')
        
        for m in valid_moves:
            q = self.q_table[idiom_id].get(m, 0.0)
            if q > best_q:
                best_q = q
                best_id = m
        
        if best_id is None:
            return None, None
        
        return best_id, self.graph.dictionary.get_text(best_id)
