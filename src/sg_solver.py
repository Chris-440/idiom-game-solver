#!/usr/bin/env python3
"""
Sprague-Grundy 求解器模块
基于 SG 定理计算成语接龙的必胜/必败态
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import pickle
import os

from .idiom_graph import IdiomGraph, GameState


class SGSolver:
    """
    Sprague-Grundy 求解器
    
    核心算法：
    1. 计算每个状态的 SG 值
    2. SG = 0 表示必败态（P-position）
    3. SG > 0 表示必胜态（N-position）
    
    必胜策略：选择移动到 SG = 0 的状态
    """
    
    def __init__(self, graph: IdiomGraph, max_cache_size: int = 1000000):
        """
        初始化求解器
        
        Args:
            graph: 成语图
            max_cache_size: 最大缓存条目数（用于内存控制）
        """
        self.graph = graph
        self.max_cache_size = max_cache_size
        
        # SG 值缓存
        self.sg_cache: Dict[Tuple[Optional[int], frozenset], int] = {}
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.states_computed = 0
    
    def mex(self, s: Set[int]) -> int:
        """
        Minimum Excludant: 集合中未出现的最小非负整数
        
        Args:
            s: 整数集合
        
        Returns:
            mex 值
        """
        i = 0
        while i in s:
            i += 1
        return i
    
    def calculate_sg(self, state: GameState) -> int:
        """
        计算状态的 SG 值（带记忆化）
        
        Args:
            state: 游戏状态
        
        Returns:
            SG 值（非负整数）
        """
        state_key = state.to_key()
        
        # 检查缓存
        if state_key in self.sg_cache:
            self.cache_hits += 1
            return self.sg_cache[state_key]
        
        self.cache_misses += 1
        self.states_computed += 1
        
        # 获取合法移动
        moves = state.get_legal_moves(self.graph)
        
        # 终止状态：无法移动，SG = 0
        if not moves:
            self._add_to_cache(state_key, 0)
            return 0
        
        # 计算所有后继状态的 SG 值
        successor_sg_values: Set[int] = set()
        for move in moves:
            new_state = state.make_move(move)
            sg = self.calculate_sg(new_state)
            successor_sg_values.add(sg)
        
        # 计算 mex
        sg_value = self.mex(successor_sg_values)
        self._add_to_cache(state_key, sg_value)
        
        return sg_value
    
    def calculate_sg_initial(self) -> int:
        """
        计算初始状态（游戏开始，无 last 成语）的 SG 值
        
        Returns:
            初始状态的 SG 值
        """
        initial_state = GameState(last_idiom=None, used_set=set())
        return self.calculate_sg(initial_state)
    
    def _add_to_cache(self, key: Tuple[Optional[int], frozenset], value: int) -> None:
        """
        添加到缓存（带大小限制）
        
        Args:
            key: 状态键
            value: SG 值
        """
        # 简单的缓存淘汰策略：超过限制时清空一半
        if len(self.sg_cache) >= self.max_cache_size:
            # 保留最近计算的条目（简单策略）
            keys_to_remove = list(self.sg_cache.keys())[:self.max_cache_size // 2]
            for k in keys_to_remove:
                del self.sg_cache[k]
        
        self.sg_cache[key] = value
    
    def is_winning_state(self, state: GameState) -> Tuple[bool, int]:
        """
        判断状态是否必胜
        
        Args:
            state: 游戏状态
        
        Returns:
            (是否必胜, SG值)
        """
        sg = self.calculate_sg(state)
        return (sg > 0, sg)
    
    def find_best_move(self, state: GameState) -> Optional[Tuple[int, str]]:
        """
        找到最优下一步
        
        Args:
            state: 当前状态
        
        Returns:
            (成语ID, 成语文本) 或 None（无合法移动）
        """
        moves = state.get_legal_moves(self.graph)
        
        if not moves:
            return None
        
        # 必胜策略：选择移动到 SG = 0 的状态
        for move in moves:
            new_state = state.make_move(move)
            sg = self.calculate_sg(new_state)
            if sg == 0:
                text = self.graph.dictionary.get_text(move)
                return (move, text)
        
        # 无必胜策略，返回第一个合法移动
        text = self.graph.dictionary.get_text(moves[0])
        return (moves[0], text)
    
    def analyze_all_initial_moves(self) -> Dict[int, Tuple[str, int, bool]]:
        """
        分析所有开局成语的优劣
        
        Returns:
            {成语ID: (成语文本, SG值, 是否必胜开局)}
        """
        results = {}
        initial_state = GameState(last_idiom=None, used_set=set())
        
        for idiom_id in self.graph.dictionary.get_all_ids():
            new_state = initial_state.make_move(idiom_id)
            sg = self.calculate_sg(new_state)
            text = self.graph.dictionary.get_text(idiom_id)
            is_winning_for_opponent = sg > 0  # 对手的视角
            
            results[idiom_id] = (text, sg, not is_winning_for_opponent)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            统计数据
        """
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) \
            if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'cache_size': len(self.sg_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': round(hit_rate, 4),
            'states_computed': self.states_computed,
        }
    
    def save_cache(self, filepath: str) -> None:
        """
        保存缓存到文件
        
        Args:
            filepath: 文件路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'sg_cache': self.sg_cache,
                'stats': {
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'states_computed': self.states_computed,
                }
            }, f)
    
    def load_cache(self, filepath: str) -> None:
        """
        从文件加载缓存
        
        Args:
            filepath: 文件路径
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.sg_cache = data['sg_cache']
                self.cache_hits = data['stats']['cache_hits']
                self.cache_misses = data['stats']['cache_misses']
                self.states_computed = data['stats']['states_computed']


class TailGroupedSolver:
    """
    按尾字分组的求解器（优化版）
    
    核心思想：
    - 不同尾字分组的成语相对独立
    - 可以分组计算，降低状态空间
    - 适合大规模成语数据
    """
    
    def __init__(self, graph: IdiomGraph):
        """
        初始化分组求解器
        
        Args:
            graph: 成语图
        """
        self.graph = graph
        self.dictionary = graph.dictionary
        
        # 按尾字分组
        self.tail_groups: Dict[str, List[int]] = defaultdict(list)
        for idiom_id in self.dictionary.get_all_ids():
            text = self.dictionary.get_text(idiom_id)
            tail_char = text[-1]
            self.tail_groups[tail_char].append(idiom_id)
        
        # 每个分组的求解器
        self.group_solvers: Dict[str, SGSolver] = {}
    
    def solve_group(self, tail_char: str) -> Dict:
        """
        求解单个尾字分组
        
        Args:
            tail_char: 尾字
        
        Returns:
            分组求解结果
        """
        group_ids = self.tail_groups[tail_char]
        
        # 创建子图（只包含该分组的成语）
        # 注意：这里需要考虑跨分组的边
        solver = SGSolver(self.graph, max_cache_size=100000)
        
        results = {}
        for idiom_id in group_ids:
            # 计算以该成语结尾时的状态
            state = GameState(last_idiom=idiom_id, used_set=set())
            sg = solver.calculate_sg(state)
            text = self.dictionary.get_text(idiom_id)
            results[idiom_id] = {
                'text': text,
                'sg': sg,
                'is_winning': sg > 0,
            }
        
        self.group_solvers[tail_char] = solver
        return results
    
    def solve_all_groups(self) -> Dict[str, Dict]:
        """
        求解所有分组
        
        Returns:
            所有分组的求解结果
        """
        all_results = {}
        for tail_char in self.tail_groups:
            all_results[tail_char] = self.solve_group(tail_char)
        return all_results
    
    def get_group_stats(self) -> Dict:
        """
        获取分组统计信息
        
        Returns:
            统计数据
        """
        group_sizes = [(char, len(ids)) for char, ids in self.tail_groups.items()]
        sorted_groups = sorted(group_sizes, key=lambda x: -x[1])
        
        return {
            'num_groups': len(self.tail_groups),
            'total_idioms': sum(len(ids) for ids in self.tail_groups.values()),
            'avg_group_size': sum(len(ids) for ids in self.tail_groups.values()) / len(self.tail_groups),
            'max_group': sorted_groups[0] if sorted_groups else ('', 0),
            'min_group': sorted_groups[-1] if sorted_groups else ('', 0),
            'top_10_groups': sorted_groups[:10],
        }


def simulate_game(solver: SGSolver, graph: IdiomGraph, 
                   max_moves: int = 100) -> Tuple[List[int], bool]:
    """
    模拟一局游戏（最优策略）
    
    Args:
        solver: SG求解器
        graph: 成语图
        max_moves: 最大移动次数
    
    Returns:
        (成语序列, 先手是否获胜)
    """
    state = GameState()
    moves_sequence = []
    current_player = 0  # 0: 先手, 1: 后手
    
    for _ in range(max_moves):
        best_move = solver.find_best_move(state)
        
        if best_move is None:
            # 无法移动，当前玩家输
            winner = 1 - current_player
            return (moves_sequence, winner == 0)
        
        idiom_id, text = best_move
        moves_sequence.append(idiom_id)
        state = state.make_move(idiom_id)
        current_player = 1 - current_player
    
    # 超过最大移动次数，视为平局（先手不输）
    return (moves_sequence, True)


if __name__ == "__main__":
    from .idiom_data import create_sample_data, IdiomDictionary
    from .idiom_graph import IdiomGraph
    
    # 测试求解器
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(30))
    
    graph = IdiomGraph(dict_obj)
    solver = SGSolver(graph)
    
    # 计算初始状态 SG 值
    initial_sg = solver.calculate_sg_initial()
    print(f"初始状态 SG 值: {initial_sg}")
    print(f"先手必胜: {initial_sg > 0}")
    
    # 找最优开局
    best_move = solver.find_best_move(GameState())
    if best_move:
        print(f"最优开局: {best_move[1]}")
    
    # 缓存统计
    print(f"\n缓存统计: {solver.get_cache_stats()}")
