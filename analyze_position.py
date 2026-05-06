"""Analyze per-starting-idiom win rates: model vs Q-table, alternating sides.
Efficient batched parallel game execution."""

import torch
import numpy as np
import sys
import pickle
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from src.rl.config import RLConfig
from src.rl.model import PolicyValueNet
from src.rl.data_preparation import load_and_index, validate_data
from src.rl.environment import IdiomGame
from src.rl.rollout import prepare_batch_input


def load_model(config, data, checkpoint_path):
    model = PolicyValueNet(
        n_idioms=data['n_idioms'], n_chars=data['n_chars'],
        idiom_dim=config.idiom_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, encoder_type=config.encoder_type,
        embedding_type=config.embedding_type,
    )
    model.idiom_emb.set_idiom_chars(data['idiom_chars'])
    model.to(config.device)
    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def load_qtable(config):
    qt_path = os.path.join(config.ckpt_dir, 'qtable.pkl')
    with open(qt_path, 'rb') as f:
        return pickle.load(f)


def make_qt_policy(qt_data):
    q = qt_data['q']
    def policy(game_state, legal_actions):
        best_a, best_v = None, -1e9
        for a in legal_actions:
            a = int(a)
            val = q[game_state['current']].get(a, 0.0)
            if val > best_v:
                best_v = val
                best_a = a
        return best_a if best_a is not None else int(legal_actions[0])
    return policy


def batched_play_vs_qt(model, qt_policy, games, model_players, config, device):
    """Run model vs Q-table games in parallel with batched inference.
    Returns list of (winner, model_won).
    """
    n = len(games)
    active = list(range(n))
    results = [None] * n

    step = 0
    while active and step < 500:
        active_games = [games[i] for i in active]

        # Collect model-turn and qt-turn indices
        model_indices = []  # local indices
        qt_indices = []

        for local_idx, orig_idx in enumerate(active):
            g = active_games[local_idx]
            if g.done:
                continue
            legal = g.get_legal_actions()
            if len(legal) == 0:
                # No moves => current player loses, opponent wins
                winner = 1 - g.current_player
                results[orig_idx] = (winner, winner == model_players[orig_idx])
                g.done = True
                continue
            if g.current_player == model_players[orig_idx]:
                model_indices.append(local_idx)
            else:
                qt_indices.append(local_idx)

        # Batched model inference
        if model_indices:
            batch_games = [active_games[i] for i in model_indices]
            batch = prepare_batch_input(batch_games, config.max_history_len,
                                        config.max_actions, device)

            with torch.no_grad():
                action_indices, _, _ = model.get_action(
                    batch['u_ids'], batch['history_ids'], batch['history_mask'],
                    batch['candidate_ids'], batch['candidate_mask'],
                    batch['player_ids'], deterministic=True,
                )

            for i, local_idx in enumerate(model_indices):
                game = active_games[local_idx]
                legal = batch['legal_actions'][i]
                action = int(legal[action_indices[i].item()])
                game.step(action)

        # Q-table turns (not batched, but Q-table lookup is O(out_degree) which is fast)
        for local_idx in qt_indices:
            game = active_games[local_idx]
            legal = game.get_legal_actions()
            action = qt_policy(game._state(), legal)
            game.step(action)

        # Update active set and record results for finished games
        new_active = []
        for orig_idx in active:
            g = games[orig_idx]
            if not g.done:
                new_active.append(orig_idx)
            elif results[orig_idx] is None:
                results[orig_idx] = (g.winner, g.winner == model_players[orig_idx])
        active = new_active
        step += 1

    # Handle truncated games (shouldn't happen with 500 step limit)
    for orig_idx in active:
        if results[orig_idx] is None:
            results[orig_idx] = (None, False)

    return results


def main():
    config = RLConfig()
    data = load_and_index(config.idiom_file)
    validate_data(data)
    adj_list = data['adj_list']
    n_idioms = data['n_idioms']
    device = config.device

    checkpoint = 'checkpoints/ckpt_iter001500.pt'
    print(f"Loading model from {checkpoint}")
    model = load_model(config, data, checkpoint)

    print("Loading Q-table...")
    qt_data = load_qtable(config)
    qt_policy = make_qt_policy(qt_data)

    rng = np.random.RandomState(42)
    valid_starts = [i for i in range(n_idioms) if len(adj_list[i]) > 0]

    # ---- Part 1: Q-Table vs Q-Table (batched, but QT is fast on CPU) ----
    n_qt = 2000
    qt_starts = rng.choice(valid_starts, size=n_qt, replace=False).tolist()
    print(f"\nPart 1: QT vs QT ({n_qt} positions)")
    # QT vs QT is deterministic, just run sequentially (QT lookup is O(1) amortized)
    qt_p0_wins = 0
    for i, sid in enumerate(qt_starts):
        g = IdiomGame(adj_list, n_idioms)
        g.reset(start_idiom=int(sid))
        while not g.done:
            legal = g.get_legal_actions()
            if len(legal) == 0:
                break
            g.step(qt_policy(g._state(), legal))
        if g.winner == 0:
            qt_p0_wins += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_qt}: P0={qt_p0_wins/(i+1):.3f}")

    print(f"  QT-QT: P0 wins {qt_p0_wins}/{n_qt} = {qt_p0_wins/n_qt:.3f}")
    print(f"  First-mover advantage: {qt_p0_wins - (n_qt - qt_p0_wins)} more P0 wins")

    # ---- Part 2: Model vs Q-Table ----
    n_samples = 3000
    n_games_per_side = 5  # multiple games to account for QT determinism + model stochasticity
    starts = rng.choice(valid_starts, size=n_samples, replace=False)

    chunk_size = 256  # parallel games per chunk
    chunk_starts = list(range(0, n_samples, chunk_size))

    # Results: start_id -> {'p0_wins': int, 'p1_wins': int, 'p0_total': int, 'p1_total': int}
    all_results = {}

    for round_idx in range(n_games_per_side):
        if round_idx > 0:
            # Shuffle starts for subsequent rounds
            starts = rng.permutation(starts)

        for side in [0, 1]:  # 0 = Model P0, 1 = Model P1
            side_label = "P0" if side == 0 else "P1"
            total_wins = 0
            total_games = 0

            for chunk_start in chunk_starts:
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_ids = starts[chunk_start:chunk_end]
                n_chunk = len(chunk_ids)

                games = []
                model_players = []
                for sid in chunk_ids:
                    g = IdiomGame(adj_list, n_idioms)
                    g.reset(start_idiom=int(sid))
                    games.append(g)
                    model_players.append(side)

                results_batch = batched_play_vs_qt(
                    model, qt_policy, games, model_players, config, device)

                for i, sid in enumerate(chunk_ids):
                    sid = int(sid)
                    _, model_won = results_batch[i]
                    if sid not in all_results:
                        all_results[sid] = {'p0_wins': 0, 'p1_wins': 0,
                                           'p0_total': 0, 'p1_total': 0}
                    if side == 0:
                        all_results[sid]['p0_total'] += 1
                        if model_won:
                            all_results[sid]['p0_wins'] += 1
                            total_wins += 1
                    else:
                        all_results[sid]['p1_total'] += 1
                        if model_won:
                            all_results[sid]['p1_wins'] += 1
                            total_wins += 1
                    total_games += 1

            total_rate = total_wins / total_games if total_games > 0 else 0
            print(f"  Round {round_idx+1}, Model {side_label}: {total_wins}/{total_games} = {total_rate:.3f}")

    # ---- Summary ----
    p0_rates = []
    p1_rates = []
    both_win = 0
    both_lose = 0
    only_p0 = 0
    only_p1 = 0

    for sid, r in all_results.items():
        p0_r = r['p0_wins'] / r['p0_total']
        p1_r = r['p1_wins'] / r['p1_total']
        p0_rates.append(p0_r)
        p1_rates.append(p1_r)

        p0_all_win = (r['p0_wins'] == r['p0_total'])
        p1_all_win = (r['p1_wins'] == r['p1_total'])
        p0_all_lose = (r['p0_wins'] == 0)
        p1_all_lose = (r['p1_wins'] == 0)

        if p0_all_win and p1_all_win:
            both_win += 1
        elif p0_all_lose and p1_all_lose:
            both_lose += 1
        elif p0_all_win and p1_all_lose:
            only_p0 += 1
        elif p0_all_lose and p1_all_win:
            only_p1 += 1

    print(f"\n{'='*60}")
    print(f"RESULTS ({n_samples} positions × {n_games_per_side} games per side)")
    print(f"{'='*60}")
    print(f"Model P0 win rate: {np.mean(p0_rates):.4f} ± {np.std(p0_rates):.4f}")
    print(f"Model P1 win rate: {np.mean(p1_rates):.4f} ± {np.std(p1_rates):.4f}")
    print(f"Overall: {(np.mean(p0_rates) + np.mean(p1_rates)) / 2:.4f}")
    print(f"Both win:  {both_win} ({100*both_win/n_samples:.1f}%)")
    print(f"Both lose: {both_lose} ({100*both_lose/n_samples:.1f}%)")
    print(f"Only P0:   {only_p0} ({100*only_p0/n_samples:.1f}%)")
    print(f"Only P1:   {only_p1} ({100*only_p1/n_samples:.1f}%)")

    # Max achievable
    max_ach = 1.0 - both_lose / n_samples
    print(f"\nMax achievable (if model wins all winnable): {max_ach:.4f}")

    # Check if both_lose positions are inherently first-mover biased
    both_lose_ids = [sid for sid, r in all_results.items()
                    if r['p0_wins'] == 0 and r['p1_wins'] == 0]
    print(f"Both-lose positions: {len(both_lose_ids)}")

    # QT-QT on a subset of both_lose positions
    bl_sample = both_lose_ids[:min(100, len(both_lose_ids))]
    qt_p0_in_bl = 0
    for sid in bl_sample:
        g = IdiomGame(adj_list, n_idioms)
        g.reset(start_idiom=sid)
        while not g.done:
            legal = g.get_legal_actions()
            if len(legal) == 0:
                break
            g.step(qt_policy(g._state(), legal))
        if g.winner == 0:
            qt_p0_in_bl += 1
    print(f"  Of {len(bl_sample)} both-lose positions, QT-QT says P0 wins: {qt_p0_in_bl}/{len(bl_sample)}")


if __name__ == '__main__':
    main()
