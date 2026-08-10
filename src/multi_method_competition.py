#!/usr/bin/env python3
"""
成语接龙多方法对比与进化竞赛
"""

import sys
import os
import json
import random
from collections import defaultdict
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import IDIOM_FILE

from src.idiom_data import IdiomDictionary
from src.idiom_graph import IdiomGraph
from src.methods.value_iteration_player import ValueIterationPlayer
from src.methods.q_learning_player import QLearningPlayer
from src.methods.mcts_player import MCTSPlayer


class TournamentResult:
    def __init__(self, player1_name, player2_name, games=100):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.games = games
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0
        self.avg_length = 0
        self.total_length = 0


class IdiomTournament:
    """成语接龙竞赛框架"""
    
    def __init__(self, graph: IdiomGraph):
        self.graph = graph
        self.players = {}
        self.results = []
        self.all_nodes = list(graph.dictionary.get_all_ids())
    
    def add_player(self, name: str, player):
        self.players[name] = player
    
    def play_game(self, player1, player2, start_idiom=None) -> Tuple[int, int, int]:
        """
        运行一局游戏
        返回: (winner, length, used_count)
        winner: 0=p1赢, 1=p2赢, -1=平局
        """
        if start_idiom is None:
            current = random.choice(self.all_nodes)
        else:
            current = start_idiom
        
        used = {current}
        players = [player1, player2]
        current_player = 0
        length = 0
        
        for _ in range(1000):  # 最大1000步
            p = players[current_player]
            move = p.select_move(current, used)
            
            if move is None:
                # 当前玩家无路可走，对手赢
                winner = 1 - current_player
                return winner, length, len(used)
            
            used.add(move)
            current = move
            length += 1
            current_player = 1 - current_player
        
        return -1, length, len(used)  # 平局
    
    def run_tournament(self, player1_name: str, player2_name: str, 
                       games: int = 100) -> TournamentResult:
        """运行完整锦标赛"""
        result = TournamentResult(player1_name, player2_name, games)
        p1 = self.players[player1_name]
        p2 = self.players[player2_name]
        
        print(f"\n{'='*50}")
        print(f"比赛: {player1_name} vs {player2_name}")
        print(f"{'='*50}")
        
        for i in range(games):
            winner, length, used_count = self.play_game(p1, p2)
            result.total_length += length
            
            if winner == 0:
                result.p1_wins += 1
            elif winner == 1:
                result.p2_wins += 1
            else:
                result.draws += 1
            
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{games}: {player1_name}={result.p1_wins}, "
                      f"{player2_name}={result.p2_wins}, 平局={result.draws}")
        
        result.avg_length = result.total_length / games if games > 0 else 0
        self.results.append(result)
        return result
    
    def run_round_robin(self, games_per_match: int = 50):
        """运行循环赛（所有选手互相比赛）"""
        names = list(self.players.keys())
        print(f"\n开始循环赛: {len(names)} 位选手, {len(names)*(len(names)-1)//2} 场比赛")
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self.run_tournament(names[i], names[j], games_per_match)
    
    def print_results(self):
        """打印比赛结果"""
        print(f"\n{'='*60}")
        print("比赛结果汇总")
        print(f"{'='*60}")
        
        # 计算每位选手的总胜率
        player_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'games': 0})
        
        for result in self.results:
            player_stats[result.player1_name]['wins'] += result.p1_wins
            player_stats[result.player1_name]['losses'] += result.p2_wins
            player_stats[result.player1_name]['games'] += result.games
            
            player_stats[result.player2_name]['wins'] += result.p2_wins
            player_stats[result.player2_name]['losses'] += result.p1_wins
            player_stats[result.player2_name]['games'] += result.games
        
        print(f"{'选手':<15} | {'胜率':>8} | {'胜':>6} | {'负':>6} | {'总局数':>8}")
        print(f"{'-'*15}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
        
        # 按胜率排序
        sorted_players = sorted(player_stats.items(), 
                                key=lambda x: x[1]['wins'] / max(1, x[1]['games']), 
                                reverse=True)
        
        for name, stats in sorted_players:
            win_rate = stats['wins'] / max(1, stats['games'])
            print(f"{name:<15} | {win_rate*100:>7.1f}% | {stats['wins']:>6} | {stats['losses']:>6} | {stats['games']:>8}")


def create_players(graph: IdiomGraph) -> Dict[str, object]:
    """创建所有选手"""
    players = {}
    
    # 1. Value Iteration
    print("创建 ValueIteration 选手...")
    vi = ValueIterationPlayer(graph, iterations=100, gamma=0.99)
    players["ValueIteration"] = vi
    
    # 2. Q-Learning (训练)
    print("\n训练 Q-Learning 选手...")
    ql = QLearningPlayer.train(graph, episodes=10000, lr=0.02, gamma=0.98)
    players["QLearning"] = ql
    
    # 3. MCTS
    print("创建 MCTS 选手...")
    mcts = MCTSPlayer(graph, simulations=500, exploration=1.414)
    players["MCTS"] = mcts
    
    # 4. Random (基线)
    class RandomPlayer:
        def __init__(self, graph):
            self.graph = graph
            self.name = "Random"
            self.neighbors = {}
            for node in graph.dictionary.get_all_ids():
                self.neighbors[node] = list(graph.get_neighbors(node))
        
        def select_move(self, current, used):
            valid = [n for n in self.neighbors.get(current, []) if n not in used]
            return random.choice(valid) if valid else None
        
        def get_name(self):
            return self.name
    
    random_player = RandomPlayer(graph)
    players["Random"] = random_player
    
    return players


def run_competition(idiom_file: str, output_dir: str = 'results'):
    """运行完整竞赛"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载成语数据...")
    dict_obj = IdiomDictionary(use_pinyin=True)
    with open(idiom_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        word = item.get('word', '')
        pinyin = item.get('pinyin', '')
        if word and len(word) == 4:
            dict_obj.add_idiom(i, word, pinyin)
    
    print(f"加载完成: {len(dict_obj)} 个成语")
    
    graph = IdiomGraph(dict_obj, use_pinyin=True)
    print(f"构建图: {graph.num_nodes} 节点, {graph.num_edges} 边")
    
    # 创建选手
    players = create_players(graph)
    
    # 运行竞赛
    tournament = IdiomTournament(graph)
    for name, player in players.items():
        tournament.add_player(name, player)
    
    # 循环赛
    tournament.run_round_robin(games_per_match=100)
    
    # 打印结果
    tournament.print_results()
    
    # 保存结果
    results_data = []
    for r in tournament.results:
        results_data.append({
            'player1': r.player1_name,
            'player2': r.player2_name,
            'p1_wins': r.p1_wins,
            'p2_wins': r.p2_wins,
            'draws': r.draws,
            'avg_length': r.avg_length,
        })
    
    output_path = os.path.join(output_dir, 'competition_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果保存到: {output_path}")
    return tournament


if __name__ == "__main__":
    idiom_file = IDIOM_FILE
    run_competition(idiom_file)
