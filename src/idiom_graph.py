#!/usr/bin/env python3
"""
成语接龙有向图模块
实现成语图的数据结构和基本操作
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from .idiom_data import IdiomDictionary


class IdiomGraph:
    """
    成语接龙有向图
    
    图结构：
    - 节点：每个成语
    - 边：(c_i, c_j) 表示 c_j 可以接在 c_i 后面
    
    这是典型的有向图，用于博弈分析
    """
    
    def __init__(self, dictionary: IdiomDictionary, use_pinyin: bool = False):
        """
        构建成语图
        
        Args:
            dictionary: 成语字典
            use_pinyin: 是否允许音同匹配
        """
        self.dictionary = dictionary
        self.use_pinyin = use_pinyin
        
        # 邻接表表示
        self.adjacency: Dict[int, List[int]] = defaultdict(list)
        self.reverse_adjacency: Dict[int, List[int]] = defaultdict(list)
        
        # 构建图
        self._build_graph()
        
        # 图统计信息
        self._compute_stats()
    
    def _build_graph(self) -> None:
        """构建邻接表"""
        for idiom_id in self.dictionary.get_all_ids():
            followers = self.dictionary.get_followers(idiom_id, self.use_pinyin)
            self.adjacency[idiom_id] = followers
            
            # 构建反向邻接表（用于分析）
            for follower_id in followers:
                self.reverse_adjacency[follower_id].append(idiom_id)
    
    def _compute_stats(self) -> None:
        """计算图统计信息"""
        self.num_nodes = len(self.dictionary)
        self.num_edges = sum(len(neighbors) for neighbors in self.adjacency.values())
        
        # 计算连通分量
        self._compute_components()
        
        # 计算度分布
        self.out_degree_dist = defaultdict(int)
        self.in_degree_dist = defaultdict(int)
        
        for node, neighbors in self.adjacency.items():
            self.out_degree_dist[len(neighbors)] += 1
        
        for node, predecessors in self.reverse_adjacency.items():
            self.in_degree_dist[len(predecessors)] += 1
        
        # 平均度
        self.avg_out_degree = self.num_edges / self.num_nodes if self.num_nodes > 0 else 0
        self.avg_in_degree = self.num_edges / self.num_nodes if self.num_nodes > 0 else 0
    
    def _compute_components(self) -> None:
        """计算连通分量（弱连通）- 迭代实现避免栈溢出"""
        visited = set()
        self.components: List[Set[int]] = []
        
        for start_node in self.dictionary.get_all_ids():
            if start_node in visited:
                continue
            
            component = set()
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                
                # 正向边
                for neighbor in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
                
                # 反向边（弱连通需要考虑）
                for predecessor in self.reverse_adjacency.get(node, []):
                    if predecessor not in visited:
                        stack.append(predecessor)
            
            self.components.append(component)
        
        # 最大连通分量
        self.max_component_size = max(len(c) for c in self.components) if self.components else 0
        self.num_components = len(self.components)
    
    def get_neighbors(self, idiom_id: int) -> List[int]:
        """
        获取所有可接龙的成语（出度邻居）
        
        Args:
            idiom_id: 成语ID
        
        Returns:
            可接龙的成语ID列表
        """
        return self.adjacency.get(idiom_id, [])
    
    def get_predecessors(self, idiom_id: int) -> List[int]:
        """
        获取所有可以接到此成语的成语（入度邻居）
        
        Args:
            idiom_id: 成语ID
        
        Returns:
            前驱成语ID列表
        """
        return self.reverse_adjacency.get(idiom_id, [])
    
    def has_edge(self, from_id: int, to_id: int) -> bool:
        """判断是否存在边"""
        return to_id in self.adjacency.get(from_id, [])
    
    def get_out_degree(self, idiom_id: int) -> int:
        """获取出度"""
        return len(self.adjacency.get(idiom_id, []))
    
    def get_in_degree(self, idiom_id: int) -> int:
        """获取入度"""
        return len(self.reverse_adjacency.get(idiom_id, []))
    
    def find_dead_ends(self) -> List[int]:
        """
        找出所有"死胡同"成语（出度为0）
        使用这些成语后，对手无法接龙
        
        Returns:
            死胡同成语ID列表
        """
        return [node for node in self.dictionary.get_all_ids() 
                if self.get_out_degree(node) == 0]
    
    def find_starters(self) -> List[int]:
        """
        找出所有可以作为开局的成语（入度为0）
        
        Returns:
            开局成语ID列表
        """
        return [node for node in self.dictionary.get_all_ids() 
                if self.get_in_degree(node) == 0]
    
    def find_bridges(self) -> List[Tuple[int, int]]:
        """
        找出所有"桥梁"成语（入度和出度都很高）
        这些成语在接龙链中起关键作用
        
        Returns:
            桥梁成语ID列表
        """
        threshold = max(5, int(self.avg_out_degree * 2))
        return [(node, self.get_in_degree(node), self.get_out_degree(node))
                for node in self.dictionary.get_all_ids()
                if self.get_in_degree(node) >= threshold 
                and self.get_out_degree(node) >= threshold]
    
    def get_stats(self) -> Dict:
        """
        获取图统计信息
        
        Returns:
            统计数据字典
        """
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'avg_out_degree': round(self.avg_out_degree, 2),
            'avg_in_degree': round(self.avg_in_degree, 2),
            'num_components': self.num_components,
            'max_component_size': self.max_component_size,
            'num_dead_ends': len(self.find_dead_ends()),
            'num_starters': len(self.find_starters()),
            'out_degree_distribution': dict(sorted(self.out_degree_dist.items())[:10]),
            'in_degree_distribution': dict(sorted(self.in_degree_dist.items())[:10]),
        }
    
    def analyze_tail_groups(self) -> Dict[str, Dict]:
        """
        分析按尾字分组的结构
        
        Returns:
            分组统计信息
        """
        groups = {}
        
        for tail_char in self.dictionary.get_tail_chars():
            group_ids = self.dictionary.get_by_tail(tail_char)
            
            # 计算该组的连通性
            group_out_degrees = [self.get_out_degree(id_) for id_ in group_ids]
            avg_out = sum(group_out_degrees) / len(group_ids) if group_ids else 0
            
            groups[tail_char] = {
                'count': len(group_ids),
                'avg_out_degree': round(avg_out, 2),
                'max_out_degree': max(group_out_degrees) if group_out_degrees else 0,
                'min_out_degree': min(group_out_degrees) if group_out_degrees else 0,
                'idioms': [self.dictionary.get_text(id_) for id_ in group_ids[:5]],  # 示例
            }
        
        return groups
    
    def __len__(self) -> int:
        return self.num_nodes
    
    def __repr__(self) -> str:
        return f"IdiomGraph(nodes={self.num_nodes}, edges={self.num_edges})"


class GameState:
    """
    游戏状态表示
    
    状态 = (last_idiom, used_set)
    - last_idiom: 最后一个说出的成语（None表示游戏开始）
    - used_set: 已使用的成语集合
    """
    
    def __init__(self, last_idiom: Optional[int] = None, 
                 used_set: Optional[Set[int]] = None):
        self.last_idiom = last_idiom
        self.used_set = used_set if used_set is not None else set()
    
    def make_move(self, idiom_id: int) -> 'GameState':
        """
        执行移动，返回新状态
        
        Args:
            idiom_id: 要使用的成语ID
        
        Returns:
            新的游戏状态
        """
        new_used = self.used_set | {idiom_id}
        return GameState(last_idiom=idiom_id, used_set=new_used)
    
    def get_legal_moves(self, graph: IdiomGraph) -> List[int]:
        """
        获取所有合法移动
        
        Args:
            graph: 成语图
        
        Returns:
            合法移动列表
        """
        if self.last_idiom is None:
            # 游戏开始，可以选择任意成语
            return [id_ for id_ in graph.dictionary.get_all_ids() 
                    if id_ not in self.used_set]
        else:
            # 必须接龙
            followers = graph.get_neighbors(self.last_idiom)
            return [id_ for id_ in followers if id_ not in self.used_set]
    
    def is_terminal(self, graph: IdiomGraph) -> bool:
        """
        判断是否为终止状态
        
        Args:
            graph: 成语图
        
        Returns:
            是否终止
        """
        return len(self.get_legal_moves(graph)) == 0
    
    def to_key(self) -> Tuple[Optional[int], frozenset]:
        """
        生成缓存键
        
        Returns:
            可哈希的状态键
        """
        return (self.last_idiom, frozenset(self.used_set))
    
    def __hash__(self) -> int:
        return hash(self.to_key())
    
    def __eq__(self, other: 'GameState') -> bool:
        return self.last_idiom == other.last_idiom and self.used_set == other.used_set
    
    def __repr__(self) -> str:
        if self.last_idiom is None:
            return f"GameState(start, used={len(self.used_set)})"
        return f"GameState(last={self.last_idiom}, used={len(self.used_set)})"


if __name__ == "__main__":
    from .idiom_data import create_sample_data, IdiomDictionary
    
    # 测试图构建
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(50))
    
    graph = IdiomGraph(dict_obj)
    
    print(f"图统计: {graph.get_stats()}")
    print(f"\n死胡同成语: {len(graph.find_dead_ends())}")
    print(f"开局成语: {len(graph.find_starters())}")
    
    # 测试游戏状态
    state = GameState()
    moves = state.get_legal_moves(graph)
    print(f"\n初始状态合法移动数: {len(moves)}")