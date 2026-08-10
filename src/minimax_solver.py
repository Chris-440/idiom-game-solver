#!/usr/bin/env python3
"""
Minimax + Alpha-Beta 剪枝求解器
用于成语接龙博弈的精确求解
"""

from typing import Dict, List, Tuple, Optional
import time

from .idiom_graph import IdiomGraph, GameState


class MinimaxSolver:
    """
    Minimax 求解器（带 Alpha-Beta 剪枝）
    
    博弈值定义：
    - +1: 先手必胜
    - -1: 先手必败
    - 0: 平局（理论上不会出现，因为游戏必然有终止）
    """
    
    def __init__(self, graph: IdiomGraph, max_cache_size: int = 1000000):
        """
        初始化求解器
        
        Args:
            graph: 成语图
            max_cache_size: 最大缓存条目数
        """
        self.graph = graph
        self.max_cache_size = max_cache_size
        
        # 博弈值缓存
        self.value_cache: Dict[Tuple[Optional[int], frozenset], int] = {}
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.nodes_evaluated = 0
        self.alpha_cutoffs = 0
        self.beta_cutoffs = 0
    
    def minimax(self, state: GameState, depth: int = 0, 
                alpha: float = -float('inf'), beta: float = float('inf'),
                is_maximizing: bool = True) -> int:
        """
        Minimax + Alpha-Beta 剪枝算法
        
        Args:
            state: 游戏状态
            depth: 当前深度
            alpha: MAX 的最佳值下界
            beta: MIN 的最佳值上界
            is_maximizing: 当前是否为先手方（MAX）
        
        Returns:
            博弈值：+1（先手胜）或 -1（先手败）
        """
        state_key = state.to_key()
        
        # 检查缓存
        if state_key in self.value_cache:
            self.cache_hits += 1
            return self.value_cache[state_key]
        
        self.cache_misses += 1
        self.nodes_evaluated += 1
        
        # 获取合法移动
        moves = state.get_legal_moves(self.graph)
        
        # 终止状态：无法移动者输
        if not moves:
            value = -1 if is_maximizing else 1
            self._add_to_cache(state_key, value)
            return value
        
        # 按启发式排序（提高剪枝效率）
        moves = self._order_moves(state, moves, is_maximizing)
        
        if is_maximizing:
            # MAX 层：找最大值
            best_value = -float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value = self.minimax(new_state, depth + 1, alpha, beta, False)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                
                if alpha >= beta:
                    self.beta_cutoffs += 1
                    break  # Beta 剪枝
            
            self._add_to_cache(state_key, best_value)
            return best_value
        else:
            # MIN 层：找最小值
            best_value = float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value = self.minimax(new_state, depth + 1, alpha, beta, True)
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                
                if beta <= alpha:
                    self.alpha_cutoffs += 1
                    break  # Alpha 剪枝
            
            self._add_to_cache(state_key, best_value)
            return best_value
    
    def _order_moves(self, state: GameState, moves: List[int], 
                     is_maximizing: bool) -> List[int]:
        """
        移动排序（启发式优化）
        
        策略：
        - MAX 层：优先选择出度小的成语（更容易让对手陷入困境）
        - MIN 层：优先选择出度大的成语（给对手更多选择）
        
        Args:
            state: 当前状态
            moves: 合法移动列表
            is_maximizing: 是否为 MAX 层
        
        Returns:
            排序后的移动列表
        """
        # 计算每个移动的启发式值
        move_scores = []
        for move in moves:
            out_degree = self.graph.get_out_degree(move)
            # MAX 层偏好低出度（让对手选择少）
            # MIN 层偏好高出度（给对手选择多）
            score = -out_degree if is_maximizing else out_degree
            move_scores.append((score, move))
        
        # 按分数排序
        move_scores.sort(key=lambda x: x[0])
        return [move for _, move in move_scores]
    
    def _add_to_cache(self, key: Tuple[Optional[int], frozenset], value: int) -> None:
        """添加到缓存（带大小限制）"""
        if len(self.value_cache) >= self.max_cache_size:
            # 简单淘汰策略
            keys_to_remove = list(self.value_cache.keys())[:self.max_cache_size // 2]
            for k in keys_to_remove:
                del self.value_cache[k]
        
        self.value_cache[key] = value
    
    def is_winning(self, state: GameState) -> Tuple[bool, int]:
        """
        判断状态是否必胜
        
        Args:
            state: 游戏状态
        
        Returns:
            (是否必胜, 博弈值)
        """
        value = self.minimax(state)
        return (value == 1, value)
    
    def find_best_move(self, state: GameState) -> Optional[Tuple[int, str]]:
        """
        找到最优下一步
        
        Args:
            state: 当前状态
        
        Returns:
            (成语ID, 成语文本) 或 None
        """
        moves = state.get_legal_moves(self.graph)
        
        if not moves:
            return None
        
        # 找到博弈值最优的移动
        best_move = None
        best_value = -float('inf') if state.last_idiom is None else -float('inf')
        
        for move in moves:
            new_state = state.make_move(move)
            value = self.minimax(new_state, is_maximizing=False)
            
            # 先手视角：找最大值
            if value > best_value:
                best_value = value
                best_move = move
        
        if best_move is not None:
            text = self.graph.dictionary.get_text(best_move)
            return (best_move, text)
        
        return (moves[0], self.graph.dictionary.get_text(moves[0]))
    
    def analyze_position(self, state: GameState) -> Dict:
        """
        分析当前局面
        
        Args:
            state: 游戏状态
        
        Returns:
            分析结果字典
        """
        moves = state.get_legal_moves(self.graph)
        
        if not moves:
            return {
                'is_terminal': True,
                'winner': 'previous' if state.last_idiom is not None else 'none',
                'legal_moves': [],
            }
        
        # 分析每个合法移动
        move_analysis = []
        for move in moves:
            new_state = state.make_move(move)
            value = self.minimax(new_state, is_maximizing=False)
            text = self.graph.dictionary.get_text(move)
            
            move_analysis.append({
                'idiom_id': move,
                'text': text,
                'value': value,
                'is_winning_for_current': value == -1,  # 对手视角
            })
        
        # 排序
        move_analysis.sort(key=lambda x: x['value'])
        
        return {
            'is_terminal': False,
            'legal_moves': move_analysis,
            'best_move': move_analysis[0] if move_analysis else None,
            'total_moves': len(moves),
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) \
            if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'cache_size': len(self.value_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': round(hit_rate, 4),
            'nodes_evaluated': self.nodes_evaluated,
            'alpha_cutoffs': self.alpha_cutoffs,
            'beta_cutoffs': self.beta_cutoffs,
            'total_cutoffs': self.alpha_cutoffs + self.beta_cutoffs,
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.cache_hits = 0
        self.cache_misses = 0
        self.nodes_evaluated = 0
        self.alpha_cutoffs = 0
        self.beta_cutoffs = 0


class IterativeDeepeningSolver:
    """
    迭代加深求解器
    
    适用于大规模问题，可以在时间限制内返回最佳结果
    """
    
    def __init__(self, graph: IdiomGraph, time_limit: float = 60.0):
        """
        初始化求解器
        
        Args:
            graph: 成语图
            time_limit: 时间限制（秒）
        """
        self.graph = graph
        self.time_limit = time_limit
        self.minimax_solver = MinimaxSolver(graph)
    
    def solve_with_timeout(self, state: GameState) -> Tuple[Optional[int], Dict]:
        """
        在时间限制内求解
        
        Args:
            state: 游戏状态
        
        Returns:
            (最佳移动ID, 统计信息)
        """
        start_time = time.time()
        moves = state.get_legal_moves(self.graph)
        
        if not moves:
            return (None, {'status': 'terminal', 'time_used': 0})
        
        best_move = moves[0]
        best_value = -float('inf')
        depth_reached = 0
        
        # 迭代加深
        for depth in range(1, len(moves) + 1):
            if time.time() - start_time > self.time_limit:
                break
            
            depth_reached = depth
            
            for move in moves:
                new_state = state.make_move(move)
                
                # 限制深度搜索
                value = self._limited_depth_search(
                    new_state, depth, -float('inf'), float('inf'), False
                )
                
                if value > best_value:
                    best_value = value
                    best_move = move
        
        time_used = time.time() - start_time
        
        return (best_move, {
            'status': 'completed' if depth_reached >= len(moves) else 'timeout',
            'depth_reached': depth_reached,
            'time_used': round(time_used, 3),
            'best_value': best_value,
            'stats': self.minimax_solver.get_stats(),
        })
    
    def _limited_depth_search(self, state: GameState, max_depth: int,
                              alpha: float, beta: float,
                              is_maximizing: bool) -> int:
        """
        深度限制的搜索
        
        Args:
            state: 游戏状态
            max_depth: 最大深度
            alpha: Alpha 值
            beta: Beta 值
            is_maximizing: 是否为 MAX 层
        
        Returns:
            博弈值
        """
        if max_depth <= 0:
            # 达到深度限制，使用启发式评估
            return self._heuristic_eval(state)
        
        state_key = state.to_key()
        
        if state_key in self.minimax_solver.value_cache:
            return self.minimax_solver.value_cache[state_key]
        
        moves = state.get_legal_moves(self.graph)
        
        if not moves:
            value = -1 if is_maximizing else 1
            self.minimax_solver._add_to_cache(state_key, value)
            return value
        
        if is_maximizing:
            best_value = -float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value = self._limited_depth_search(
                    new_state, max_depth - 1, alpha, beta, False
                )
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            self.minimax_solver._add_to_cache(state_key, best_value)
            return best_value
        else:
            best_value = float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value = self._limited_depth_search(
                    new_state, max_depth - 1, alpha, beta, True
                )
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            self.minimax_solver._add_to_cache(state_key, best_value)
            return best_value
    
    def _heuristic_eval(self, state: GameState) -> int:
        """
        启发式评估函数
        
        简单策略：
        - 出度越多越有利
        - 已使用成语越少越有利
        
        Args:
            state: 游戏状态
        
        Returns:
            评估值（-1 到 1）
        """
        moves = state.get_legal_moves(self.graph)
        
        if not moves:
            return -1  # 无法移动
        
        # 平均出度
        avg_out = sum(self.graph.get_out_degree(m) for m in moves) / len(moves)
        
        # 简单归一化
        max_possible_out = max(self.graph.get_out_degree(m) for m in 
                               self.graph.dictionary.get_all_ids()) or 1
        
        score = avg_out / max_possible_out
        
        # 转换为博弈值范围
        if score > 0.5:
            return 1
        elif score < 0.3:
            return -1
        else:
            return 0


if __name__ == "__main__":
    from .idiom_data import create_sample_data, IdiomDictionary
    from .idiom_graph import IdiomGraph
    
    # 测试求解器
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(30))
    
    graph = IdiomGraph(dict_obj)
    solver = MinimaxSolver(graph)
    
    # 测试初始状态
    initial_state = GameState()
    is_win, value = solver.is_winning(initial_state)
    
    print(f"初始状态博弈值: {value}")
    print(f"先手必胜: {is_win}")
    
    # 找最优第一步
    best = solver.find_best_move(initial_state)
    if best:
        print(f"最优第一步: {best[1]}")
    
    # 统计信息
    print(f"\n求解统计: {solver.get_stats()}")
    
    # 测试迭代加深
    print("\n--- 迭代加深测试 ---")
    id_solver = IterativeDeepeningSolver(graph, time_limit=5.0)
    result = id_solver.solve_with_timeout(initial_state)
    
    if result[0]:
        print(f"最佳移动: {dict_obj.get_text(result[0])}")
    print(f"统计: {result[1]}")
