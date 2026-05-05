import numpy as np


class IdiomGame:
    """Single game of idiom solitaire. Uses numpy boolean mask for O(1) used-set lookup."""

    def __init__(self, adj_list, n_idioms, start_pool=None):
        self.adj_list = adj_list
        self.n_idioms = n_idioms
        self._used_mask = np.zeros(n_idioms, dtype=bool)

        if start_pool is None:
            self.start_pool = np.array(
                [i for i in range(n_idioms) if len(adj_list[i]) > 0],
                dtype=np.int32
            )
        else:
            self.start_pool = start_pool

    def reset(self, start_idiom=None):
        if start_idiom is None:
            start_idiom = int(np.random.choice(self.start_pool))
        self.current = start_idiom
        self.history = [start_idiom]
        self._used_mask.fill(False)
        self._used_mask[start_idiom] = True
        self.current_player = 0
        self.done = False
        self.winner = None
        self.n_steps = 0
        return self._state()

    def get_legal_actions(self):
        if self.done:
            return np.array([], dtype=np.int32)
        candidates = self.adj_list[self.current]
        if len(candidates) == 0:
            return np.array([], dtype=np.int32)
        legal = candidates[~self._used_mask[candidates]]
        return legal

    def step(self, action):
        assert not self.done, "Game already ended"
        self.current = action
        self.history.append(action)
        self._used_mask[action] = True
        self.n_steps += 1
        self.current_player = 1 - self.current_player

        # Check if next player has legal moves
        next_candidates = self.adj_list[self.current]
        if len(next_candidates) == 0 or self._used_mask[next_candidates].all():
            self.done = True
            self.winner = 1 - self.current_player

        return self._state()

    def _state(self):
        return {
            'current': self.current,
            'history': self.history,
            'history_len': len(self.history),
            'current_player': self.current_player,
            'done': self.done,
            'winner': self.winner,
            'n_steps': self.n_steps,
        }


def analyze_start_nodes(adj_list, n_idioms, n_samples=10000):
    """Analyze impact of start node filtering on game quality."""
    game = IdiomGame(adj_list, n_idioms,
                     start_pool=np.arange(n_idioms, dtype=np.int32))
    lengths_uniform = []
    for _ in range(n_samples):
        game.reset()
        while not game.done:
            legal = game.get_legal_actions()
            if len(legal) == 0:
                break
            game.step(int(np.random.choice(legal)))
        lengths_uniform.append(game.n_steps)

    game2 = IdiomGame(adj_list, n_idioms)
    lengths_filtered = []
    for _ in range(n_samples):
        game2.reset()
        while not game2.done:
            legal = game2.get_legal_actions()
            if len(legal) == 0:
                break
            game2.step(int(np.random.choice(legal)))
        lengths_filtered.append(game2.n_steps)

    print(f"Uniform start - mean steps: {np.mean(lengths_uniform):.1f}, "
          f"0-step ratio: {sum(1 for l in lengths_uniform if l == 0) / n_samples:.1%}")
    print(f"Filtered start - mean steps: {np.mean(lengths_filtered):.1f}, "
          f"0-step ratio: {sum(1 for l in lengths_filtered if l == 0) / n_samples:.1%}")

    p99 = int(np.percentile(lengths_filtered, 99))
    print(f"99th percentile steps: {p99} -> suggested max_history_len = {p99}")
    return p99


def test_environment():
    """Environment logic correctness tests."""
    # Test 1: 3-node cycle
    adj = [np.array([1]), np.array([2]), np.array([0])]
    game = IdiomGame(adj, 3, start_pool=np.array([0]))
    state = game.reset(start_idiom=0)

    assert state['current'] == 0
    assert state['current_player'] == 0
    assert not state['done']

    legal = game.get_legal_actions()
    assert list(legal) == [1], f"Expected [1], got {legal}"

    state = game.step(1)
    assert state['current'] == 1
    assert state['current_player'] == 1
    assert not state['done']

    state = game.step(2)
    assert state['current'] == 2
    assert state['current_player'] == 0

    legal = game.get_legal_actions()
    assert len(legal) == 0
    assert state['done']
    assert state['winner'] == 1

    # Test 2: Dead node
    adj2 = [np.array([1]), np.array([], dtype=np.int32)]
    game2 = IdiomGame(adj2, 2, start_pool=np.array([0]))
    game2.reset(start_idiom=0)
    game2.step(1)
    assert game2.done
    assert game2.winner == 0

    # Test 3: Used filtering
    adj3 = [np.array([1, 2]), np.array([0]), np.array([0])]
    game3 = IdiomGame(adj3, 3, start_pool=np.array([0]))
    game3.reset(start_idiom=0)
    game3.step(1)
    assert game3.done
    assert game3.winner == 0

    print("Environment tests ALL PASSED")
