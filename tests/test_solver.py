#!/usr/bin/env python3
"""
单元测试模块
测试成语接龙博弈求解器的各个组件
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idiom_data import IdiomDictionary, create_sample_data
from src.idiom_graph import IdiomGraph, GameState
from src.sg_solver import SGSolver
from src.minimax_solver import MinimaxSolver


class TestIdiomDictionary(unittest.TestCase):
    """测试成语字典类"""
    
    def setUp(self):
        """测试前准备"""
        self.dict_obj = IdiomDictionary(use_pinyin=True)
        self.dict_obj.load_from_list(create_sample_data(30))
    
    def test_load_idioms(self):
        """测试成语加载"""
        self.assertEqual(len(self.dict_obj), 30)
    
    def test_get_by_head(self):
        """测试首字索引"""
        idioms = self.dict_obj.get_by_head('一')
        self.assertTrue(len(idioms) > 0)
    
    def test_get_by_tail(self):
        """测试尾字索引"""
        idioms = self.dict_obj.get_by_tail('意')
        self.assertTrue(len(idioms) > 0)
    
    def test_get_followers(self):
        """测试接龙查询"""
        # "一心一意" 尾字是 "意"
        followers = self.dict_obj.get_followers(0)
        # 应该能找到以 "意" 开头的成语
        self.assertTrue(len(followers) > 0)
    
    def test_get_stats(self):
        """测试统计信息"""
        stats = self.dict_obj.get_stats()
        self.assertEqual(stats['total_idioms'], 30)
        self.assertTrue(stats['unique_head_chars'] > 0)
        self.assertTrue(stats['unique_tail_chars'] > 0)


class TestIdiomGraph(unittest.TestCase):
    """测试成语图类"""
    
    def setUp(self):
        """测试前准备"""
        self.dict_obj = IdiomDictionary(use_pinyin=True)
        self.dict_obj.load_from_list(create_sample_data(30))
        self.graph = IdiomGraph(self.dict_obj)
    
    def test_graph_build(self):
        """测试图构建"""
        self.assertEqual(self.graph.num_nodes, 30)
        self.assertTrue(self.graph.num_edges > 0)
    
    def test_get_neighbors(self):
        """测试邻居查询"""
        neighbors = self.graph.get_neighbors(0)
        self.assertTrue(len(neighbors) >= 0)
    
    def test_find_dead_ends(self):
        """测试死胡同查找"""
        dead_ends = self.graph.find_dead_ends()
        self.assertTrue(len(dead_ends) >= 0)
    
    def test_find_starters(self):
        """测试开局成语查找"""
        starters = self.graph.find_starters()
        self.assertTrue(len(starters) >= 0)
    
    def test_get_stats(self):
        """测试图统计"""
        stats = self.graph.get_stats()
        self.assertEqual(stats['num_nodes'], 30)
        self.assertTrue(stats['num_components'] >= 1)


class TestGameState(unittest.TestCase):
    """测试游戏状态类"""
    
    def setUp(self):
        """测试前准备"""
        self.dict_obj = IdiomDictionary(use_pinyin=True)
        self.dict_obj.load_from_list(create_sample_data(30))
        self.graph = IdiomGraph(self.dict_obj)
    
    def test_initial_state(self):
        """测试初始状态"""
        state = GameState()
        moves = state.get_legal_moves(self.graph)
        # 初始状态可以选择任意成语
        self.assertEqual(len(moves), 30)
    
    def test_make_move(self):
        """测试移动"""
        state = GameState()
        new_state = state.make_move(0)
        self.assertEqual(new_state.last_idiom, 0)
        self.assertTrue(0 in new_state.used_set)
    
    def test_legal_moves_after_move(self):
        """测试移动后的合法移动"""
        state = GameState()
        new_state = state.make_move(0)
        moves = new_state.get_legal_moves(self.graph)
        # 移动后只能接龙
        self.assertTrue(len(moves) < 30)
    
    def test_terminal_state(self):
        """测试终止状态"""
        # 使用所有成语后应为终止状态
        state = GameState(last_idiom=0, used_set=set(range(30)))
        self.assertTrue(state.is_terminal(self.graph))


class TestSGSolver(unittest.TestCase):
    """测试SG求解器"""
    
    def setUp(self):
        """测试前准备"""
        self.dict_obj = IdiomDictionary(use_pinyin=True)
        self.dict_obj.load_from_list(create_sample_data(30))
        self.graph = IdiomGraph(self.dict_obj)
        self.solver = SGSolver(self.graph)
    
    def test_calculate_sg_initial(self):
        """测试初始SG值计算"""
        sg = self.solver.calculate_sg_initial()
        self.assertTrue(sg >= 0)
    
    def test_terminal_sg_zero(self):
        """测试终止状态SG=0"""
        # 死胡同成语使用后，对手无法接龙
        dead_ends = self.graph.find_dead_ends()
        if dead_ends:
            dead_id = dead_ends[0]
            state = GameState(last_idiom=dead_id, used_set={dead_id})
            sg = self.solver.calculate_sg(state)
            self.assertEqual(sg, 0)
    
    def test_find_best_move(self):
        """测试最优移动查找"""
        state = GameState()
        best_move = self.solver.find_best_move(state)
        self.assertTrue(best_move is not None)
    
    def test_winning_strategy(self):
        """测试必胜策略"""
        initial_sg = self.solver.calculate_sg_initial()
        if initial_sg > 0:
            # 必胜态应该能找到移动到SG=0的策略
            best_move = self.solver.find_best_move(GameState())
            if best_move:
                new_state = GameState().make_move(best_move[0])
                new_sg = self.solver.calculate_sg(new_state)
                self.assertEqual(new_sg, 0)


class TestMinimaxSolver(unittest.TestCase):
    """测试Minimax求解器"""
    
    def setUp(self):
        """测试前准备"""
        self.dict_obj = IdiomDictionary(use_pinyin=True)
        self.dict_obj.load_from_list(create_sample_data(30))
        self.graph = IdiomGraph(self.dict_obj)
        self.solver = MinimaxSolver(self.graph)
    
    def test_is_winning(self):
        """测试胜负判定"""
        state = GameState()
        is_win, value = self.solver.is_winning(state)
        self.assertTrue(value in [-1, 1])
    
    def test_find_best_move(self):
        """测试最优移动"""
        state = GameState()
        best_move = self.solver.find_best_move(state)
        self.assertTrue(best_move is not None)
    
    def test_consistency_with_sg(self):
        """测试与SG求解器一致性"""
        sg_solver = SGSolver(self.graph)
        
        sg_win = sg_solver.calculate_sg_initial() > 0
        mm_win = self.solver.is_winning(GameState())[0]
        
        self.assertEqual(sg_win, mm_win)


class TestAlgorithmConsistency(unittest.TestCase):
    """测试算法一致性"""
    
    def test_sg_minimax_consistency_small(self):
        """小规模一致性测试"""
        for n in [10, 20, 30]:
            dict_obj = IdiomDictionary(use_pinyin=True)
            dict_obj.load_from_list(create_sample_data(n))
            graph = IdiomGraph(dict_obj)
            
            sg_solver = SGSolver(graph)
            mm_solver = MinimaxSolver(graph)
            
            sg_win = sg_solver.calculate_sg_initial() > 0
            mm_win = mm_solver.is_winning(GameState())[0]
            
            self.assertEqual(sg_win, mm_win, 
                f"SG和Minimax不一致: n={n}, sg={sg_win}, mm={mm_win}")
    
    def test_winning_strategy_validity(self):
        """必胜策略有效性测试"""
        dict_obj = IdiomDictionary(use_pinyin=True)
        dict_obj.load_from_list(create_sample_data(30))
        graph = IdiomGraph(dict_obj)
        solver = SGSolver(graph)
        
        initial_sg = solver.calculate_sg_initial()
        if initial_sg > 0:
            best_move = solver.find_best_move(GameState())
            if best_move:
                new_state = GameState().make_move(best_move[0])
                new_sg = solver.calculate_sg(new_state)
                self.assertEqual(new_sg, 0, 
                    "必胜策略应移动到SG=0的状态")


if __name__ == "__main__":
    unittest.main(verbosity=2)