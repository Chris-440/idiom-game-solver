#!/usr/bin/env python3
"""Train a Q-table opponent on the pruned idiom graph for use as RL baseline.

Uses numpy for speed, multi-core self-play with TD updates.
Mirrors the Go code's Q-learning approach.
"""

import sys, os, pickle, time
from multiprocessing import Pool, cpu_count
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def load_data():
    from src.rl.data_preparation import load_and_index
    return load_and_index()


def build_q_table(data):
    """Build Q-table as list of dicts (sparse, like Go version)."""
    adj = data['adj_list']
    n = data['n_idioms']
    q = []
    for i in range(n):
        d = {}
        for v in adj[i]:
            d[int(v)] = np.random.randn() * 0.01  # small random init
        q.append(d)
    return q


def play_one_game(q, adj, rng, epsilon=0.1):
    """Play one self-play game, return trajectory and winner."""
    n = len(adj)
    start = int(rng.integers(0, n))
    used = np.zeros(n, dtype=bool)
    used[start] = True
    current = start
    player = 0
    trajectory = []

    for _ in range(500):
        neighbors = adj[current]
        valid = [v for v in neighbors if not used[int(v)]]

        if len(valid) == 0:
            winner = 1 - player
            return trajectory, winner

        # Epsilon-greedy
        if rng.random() < epsilon:
            move = int(rng.choice(valid))
        else:
            best_val = -1e9
            best_move = valid[0]
            for v in valid:
                vi = int(v)
                val = q[current].get(vi, 0.0)
                if val > best_val:
                    best_val = val
                    best_move = vi
            move = best_move

        used[move] = True
        trajectory.append((player, current, move))
        current = move
        player = 1 - player

    return trajectory, -1  # draw (should be rare on pruned graph)


def update_q(q, trajectory, winner, lr=0.05, gamma=0.85):
    """TD update — reverse pass through trajectory."""
    for i in range(len(trajectory) - 1, -1, -1):
        p, s, a = trajectory[i]
        s, a = int(s), int(a)

        # Terminal reward
        if i == len(trajectory) - 1:
            r = 1.0 if p == winner else -1.0
        else:
            r = 0.0

        # Next max Q (from opponent's perspective, zero-sum)
        next_max = 0.0
        if i < len(trajectory) - 1:
            _, ns, _ = trajectory[i + 1]
            ns = int(ns)
            if q[ns]:
                next_max = max(q[ns].values())

        # Zero-sum TD: opponent's max Q is our penalty
        target = r + gamma * (-next_max)
        old = q[s].get(a, 0.0)
        q[s][a] = old + lr * (target - old)


def worker_run(args):
    """Worker process: play + update games, return merged Q-table changes."""
    worker_id, adj_flat, n_games, lr, gamma, eps_start, eps_min, seed = args

    # Rebuild adj from flat format
    adj = []
    idx = 0
    while idx < len(adj_flat):
        n_succ = int(adj_flat[idx])
        idx += 1
        succs = adj_flat[idx:idx + n_succ]
        adj.append(np.array(succs, dtype=np.int32))
        idx += n_succ
    n = len(adj)

    q = []
    for i in range(n):
        d = {}
        for v in adj[i]:
            d[int(v)] = np.random.randn() * 0.01
        q.append(d)

    rng = np.random.default_rng(seed)
    total_steps = 0
    wins = 0

    for ep in range(n_games):
        progress = ep / n_games
        eps = eps_start * (1 - progress) + eps_min
        current_lr = lr * (1.0 - progress * 0.8)  # decay to 20%

        traj, winner = play_one_game(q, adj, rng, epsilon=eps)
        if winner >= 0:
            update_q(q, traj, winner, lr=current_lr, gamma=gamma)
            total_steps += len(traj)
            if traj and traj[-1][0] == winner:
                wins += 1

    return {'q': q, 'steps': total_steps, 'wins': wins, 'worker_id': worker_id}


def merge_q_tables(q_list):
    """Average Q-tables from multiple workers."""
    n = len(q_list[0])
    merged = []
    for i in range(n):
        d = {}
        all_keys = set()
        for q in q_list:
            all_keys.update(q[i].keys())
        for k in all_keys:
            vals = [q[i].get(k, 0.0) for q in q_list]
            d[k] = sum(vals) / len(vals)
        merged.append(d)
    return merged


def evaluate_vs_random(q, adj, n_games=2000):
    """Evaluate Q-table vs random policy."""
    rng = np.random.default_rng(42)
    n = len(adj)
    wins = 0
    for i in range(n_games):
        q_player = i % 2
        start = int(rng.integers(0, n))
        used = np.zeros(n, dtype=bool)
        used[start] = True
        current = start
        player = 0

        for _ in range(500):
            neighbors = adj[current]
            valid = [v for v in neighbors if not used[int(v)]]
            if not valid:
                if 1 - player == q_player:
                    wins += 1
                break

            if player == q_player:
                best = -1e9
                move = int(valid[0])
                for v in valid:
                    vi = int(v)
                    val = q[current].get(vi, 0.0)
                    if val > best:
                        best = val
                        move = vi
            else:
                move = int(rng.choice(valid))

            used[move] = True
            current = move
            player = 1 - player

    return wins / n_games


def train_qtable(data, total_games=2_000_000, lr=0.05, gamma=0.85,
                 eps_start=0.3, eps_min=0.05, eval_interval=50000):
    """Main training loop with multi-core parallelism."""
    adj = data['adj_list']
    n = data['n_idioms']
    n_workers = min(cpu_count(), 32)

    print(f"Training Q-table: {total_games:,} games, {n_workers} workers")
    print(f"  Graph: {n} nodes, LR={lr}, gamma={gamma}")
    print()

    # Flatten adj for multiprocessing
    adj_flat = []
    for a in adj:
        adj_flat.append(len(a))
        adj_flat.extend(a.tolist())
    adj_flat = np.array(adj_flat, dtype=np.int32)

    games_per_worker = total_games // (n_workers * 10)  # 10 rounds of parallel training

    q = build_q_table(data)

    for round_idx in range(10):
        round_games = games_per_worker * n_workers
        t0 = time.time()

        seeds = [np.random.randint(0, 2**31) for _ in range(n_workers)]
        args_list = [(w, adj_flat, games_per_worker, lr, gamma,
                      eps_start, eps_min, seeds[w]) for w in range(n_workers)]

        with Pool(n_workers) as pool:
            results = pool.map(worker_run, args_list)

        # Merge Q-tables
        worker_qs = [r['q'] for r in results]
        q = merge_q_tables(worker_qs)

        total_steps = sum(r['steps'] for r in results)
        dt = time.time() - t0
        total_games_so_far = (round_idx + 1) * round_games

        # Evaluate
        wr = evaluate_vs_random(q, adj, n_games=1000)
        print(f"  Round {round_idx + 1}/10: {total_games_so_far:>10,} games "
              f"| {total_steps:>8,} steps | {dt:5.0f}s "
              f"| vs_random={wr:.1%}")

    return q


def main():
    print("=" * 60)
    print("Q-TABLE TRAINING")
    print("=" * 60)

    data = load_data()
    print(f"Graph: {data['n_idioms']} nodes after pruning")

    q = train_qtable(data, total_games=2_000_000)

    # Final evaluation
    wr = evaluate_vs_random(q, data['adj_list'], n_games=5000)
    print(f"\nFinal vs_random: {wr:.1%}")

    # Save
    cache_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'qtable.pkl')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'q': q, 'n_idioms': data['n_idioms']}, f)
    print(f"Saved to {cache_path}")

    # Also save as policy function
    from src.rl.qt_opponent import make_vi_policy
    # Convert Q-table to value-like array for policy
    values = np.zeros(data['n_idioms'], dtype=np.float32)
    for i in range(data['n_idioms']):
        if q[i]:
            values[i] = max(q[i].values())  # best Q for this node
    policy = make_vi_policy(values, data['adj_list'])
    print(f"Policy ready for RL training.")


if __name__ == '__main__':
    main()
