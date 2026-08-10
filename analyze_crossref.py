"""Cross-reference: model only_p0/only_p1 positions vs QT-QT inherent advantage."""

import torch
import numpy as np
import sys, pickle, os
sys.path.insert(0, os.path.dirname(__file__))

from src.rl.config import RLConfig
from src.rl.model import PolicyValueNet
from src.rl.data_preparation import load_and_index, validate_data
from src.rl.environment import IdiomGame
from src.rl.rollout import prepare_batch_input


def load_model(config, data, ckpt_path):
    model = PolicyValueNet(
        n_idioms=data['n_idioms'], n_chars=data['n_chars'],
        idiom_dim=config.idiom_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, encoder_type=config.encoder_type,
        embedding_type=config.embedding_type,
    )
    model.idiom_emb.set_idiom_chars(data['idiom_chars'])
    model.to(config.device)
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
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


def qt_vs_qt_outcome(qt_policy, adj_list, n_idioms, start_id):
    """Returns winner of QT-QT from given start."""
    g = IdiomGame(adj_list, n_idioms)
    g.reset(start_idiom=int(start_id))
    while not g.done:
        legal = g.get_legal_actions()
        if len(legal) == 0:
            break
        g.step(qt_policy(g._state(), legal))
    return g.winner


def play_one_game(model, qt_policy, adj_list, n_idioms, start_id, model_player, config, device):
    """Play a single model vs QT game, return model_won."""
    g = IdiomGame(adj_list, n_idioms)
    g.reset(start_idiom=int(start_id))
    while not g.done:
        legal = g.get_legal_actions()
        if len(legal) == 0:
            break
        if g.current_player == model_player:
            batch = prepare_batch_input([g], config.max_history_len, config.max_actions, device)
            with torch.no_grad():
                action_indices, _, _ = model.get_action(
                    batch['u_ids'], batch['history_ids'], batch['history_mask'],
                    batch['candidate_ids'], batch['candidate_mask'],
                    batch['player_ids'], deterministic=True,
                )
            g.step(int(legal[action_indices[0].item()]))
        else:
            g.step(qt_policy(g._state(), legal))
    return g.winner == model_player


def main():
    config = RLConfig()
    data = load_and_index(config.idiom_file)
    validate_data(data)
    adj_list = data['adj_list']
    n_idioms = data['n_idioms']
    device = config.device

    model = load_model(config, data, 'checkpoints/ckpt_iter001500.pt')
    qt_data = load_qtable(config)
    qt_policy = make_qt_policy(qt_data)

    rng = np.random.RandomState(42)
    valid_starts = [i for i in range(n_idioms) if len(adj_list[i]) > 0]
    n_samples = 2000
    starts = rng.choice(valid_starts, size=n_samples, replace=False)

    print(f"Evaluating {n_samples} positions: model P0, model P1, QT-QT...")

    only_p0_ids = []
    only_p1_ids = []
    both_win_ids = []
    both_lose_ids = []
    for i, sid in enumerate(starts):
        sid = int(sid)
        won_p0 = play_one_game(model, qt_policy, adj_list, n_idioms, sid, 0, config, device)
        won_p1 = play_one_game(model, qt_policy, adj_list, n_idioms, sid, 1, config, device)
        qt_winner = qt_vs_qt_outcome(qt_policy, adj_list, n_idioms, sid)

        if won_p0 and won_p1:
            both_win_ids.append((sid, qt_winner))
        elif won_p0 and not won_p1:
            only_p0_ids.append((sid, qt_winner))
        elif not won_p0 and won_p1:
            only_p1_ids.append((sid, qt_winner))
        elif not won_p0 and not won_p1:
            both_lose_ids.append((sid, qt_winner))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_samples}")

    print(f"\n{'='*60}")
    print(f"CROSS-REFERENCE: Model performance vs QT-QT inherent advantage")
    print(f"{'='*60}")

    def qt_winner_dist(ids_with_qt):
        if not ids_with_qt:
            return {}
        winners = [qt for _, qt in ids_with_qt]
        return {
            'P0_wins': winners.count(0),
            'P1_wins': winners.count(1),
            'None': winners.count(None),
            'P0_rate': winners.count(0) / len(winners) if winners else 0,
            'P1_rate': winners.count(1) / len(winners) if winners else 0,
        }

    print(f"\n--- Only P0 wins ({len(only_p0_ids)} positions) ---")
    if only_p0_ids:
        dist = qt_winner_dist(only_p0_ids)
        print(f"  QT-QT says P0 wins: {dist['P0_wins']} ({dist['P0_rate']:.1%})")
        print(f"  QT-QT says P1 wins: {dist['P1_wins']} ({dist.get('P1_rate', 0):.1%})")
        if dist['P0_rate'] > 0.9:
            print("  → Strong correlation with a P0 win under this Q-table policy; "
                  "not a minimax certificate.")
        else:
            print(f"  → Model's P1 strategy CAN be improved on these positions.")

    print(f"\n--- Only P1 wins ({len(only_p1_ids)} positions) ---")
    if only_p1_ids:
        dist = qt_winner_dist(only_p1_ids)
        print(f"  QT-QT says P0 wins: {dist['P0_wins']} ({dist['P0_rate']:.1%})")
        print(f"  QT-QT says P1 wins: {dist['P1_wins']} ({dist.get('P1_rate', 0):.1%})")
        if dist.get('P1_rate', 0) > 0.9:
            print("  → Strong correlation with a P1 win under this Q-table policy; "
                  "not a minimax certificate.")
        else:
            print(f"  → Model's P0 strategy CAN be improved on these positions.")

    print(f"\n--- Both win ({len(both_win_ids)} positions) ---")
    if both_win_ids:
        dist = qt_winner_dist(both_win_ids)
        print(f"  QT-QT P0 wins: {dist.get('P0_rate', 0):.1%}")

    print(f"\n--- Both lose ({len(both_lose_ids)} positions) ---")
    if both_lose_ids:
        dist = qt_winner_dist(both_lose_ids)
        print(f"  QT-QT P0 wins: {dist.get('P0_rate', 0):.1%}")

    # Summary
    total = len(starts)
    p0_win = (len(both_win_ids) + len(only_p0_ids)) / total
    p1_win = (len(both_win_ids) + len(only_p1_ids)) / total
    overall = (p0_win + p1_win) / 2
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Model P0 win rate: {p0_win:.3f}")
    print(f"Model P1 win rate: {p1_win:.3f}")
    print(f"Overall: {overall:.3f}")
    print("Scope: empirical performance against this fixed Q-table; "
          "not a game-theoretic upper bound.")


if __name__ == '__main__':
    main()
