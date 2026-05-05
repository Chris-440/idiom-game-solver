"""Value Iteration opponent policy for RL training.

Runs value iteration on the pruned idiom graph to produce a strong
baseline opponent (mirrors the Go code's VIPlayer).
"""

import numpy as np


def train_vi_policy(adj_list, n_idioms, gamma=0.99, iterations=100):
    """Value iteration on the idiom graph.

    Returns:
        values: np.array of shape (n_idioms,) — expected value from each node
    """
    values = np.zeros(n_idioms, dtype=np.float32)

    for iteration in range(iterations):
        new_values = np.zeros(n_idioms, dtype=np.float32)
        max_delta = 0.0

        for u in range(n_idioms):
            successors = adj_list[u]
            if len(successors) == 0:
                new_values[u] = -1.0
                continue

            best = -1e9
            for v in successors:
                if len(adj_list[v]) == 0:
                    val = 1.0  # opponent has no moves → instant win
                else:
                    val = -gamma * values[v]  # zero-sum perspective flip
                if val > best:
                    best = val
            new_values[u] = best
            max_delta = max(max_delta, abs(new_values[u] - values[u]))

        values = new_values

        if iteration % 20 == 0:
            print(f"  VI iter {iteration}: max delta = {max_delta:.5f}")
        if max_delta < 1e-6:
            print(f"  VI converged at iter {iteration}")
            break

    return values


def make_vi_policy(values, adj_list):
    """Create a policy function from value iteration results.

    Returns a callable with signature (game_state, legal_actions) -> action_id.
    """
    def policy(game_state, legal_actions):
        if len(legal_actions) == 0:
            return -1
        best_action = legal_actions[0]
        best_val = -1e9
        for a in legal_actions:
            v = values[a]
            if v < best_val:  # pick LOWEST opponent value (best for us)
                best_val = v
                best_action = a
        return int(best_action)

    return policy


def build_and_save(data_dir=None, force=False):
    """Load data, train VI policy, return the policy function."""
    import os, pickle

    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'checkpoints', 'vi_policy.pkl')

    if not force and os.path.exists(cache_path):
        print(f"Loading cached VI policy from {cache_path}")
        with open(cache_path, 'rb') as f:
            values = pickle.load(f)
        return values

    from src.rl.data_preparation import load_and_index
    data = load_and_index()
    adj_list = data['adj_list']
    n_idioms = data['n_idioms']

    print(f"Running value iteration on {n_idioms} nodes...")
    values = train_vi_policy(adj_list, n_idioms)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(values, f)
    print(f"VI policy saved to {cache_path}")

    return values
