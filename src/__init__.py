#!/usr/bin/env python3
"""
成语接龙博弈求解器
基于 Sprague-Grundy 定理和 Minimax 算法
"""

from .idiom_data import IdiomDictionary, create_sample_data
from .idiom_graph import IdiomGraph, GameState
from .sg_solver import SGSolver, TailGroupedSolver, simulate_game
from .minimax_solver import MinimaxSolver, IterativeDeepeningSolver
from .experiment import ExperimentRunner

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from functools import lru_cache
import pickle
import os
import sys
import time


__all__ = [
    'IdiomDictionary',
    'IdiomGraph',
    'GameState',
    'SGSolver',
    'TailGroupedSolver',
    'MinimaxSolver',
    'IterativeDeepeningSolver',
    'ExperimentRunner',
    'simulate_game',
    'create_sample_data',
]