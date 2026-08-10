import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt
from .environment import IdiomGame
from .rollout import prepare_batch_input, prepare_model_input


def evaluate_vs_random(model, adj_list, n_idioms,
                       max_history_len, max_actions,
                       n_games=1000, device='cuda'):
    """Vectorized model vs random. Batches all active games for GPU inference."""
    model.eval()
    wins = 0

    # Split into chunks to avoid OOM with too many parallel games
    chunk_size = 256
    for chunk_start in range(0, n_games, chunk_size):
        n_chunk = min(chunk_size, n_games - chunk_start)

        games = []
        for i in range(n_chunk):
            g = IdiomGame(adj_list, n_idioms)
            g.reset()
            games.append(g)
        model_players = [(chunk_start + i) % 2 for i in range(n_chunk)]
        active = list(range(n_chunk))

        while active:
            active_games = [games[i] for i in active]

            model_indices = []
            opponent_indices = []

            for local_idx, orig_idx in enumerate(active):
                g = active_games[local_idx]
                if g.current_player == model_players[orig_idx]:
                    model_indices.append(local_idx)
                else:
                    opponent_indices.append(local_idx)

            if model_indices:
                batch_games = [active_games[i] for i in model_indices]
                batch = prepare_batch_input(batch_games, max_history_len,
                                            max_actions, device)

                with torch.no_grad():
                    action_indices, _, _ = model.get_action(
                        batch['u_ids'], batch['history_ids'],
                        batch['history_mask'],
                        batch['candidate_ids'], batch['candidate_mask'],
                        batch['player_ids'], deterministic=True
                    )

                for i, local_idx in enumerate(model_indices):
                    game = active_games[local_idx]
                    legal = batch['legal_actions'][i]
                    game.step(int(legal[action_indices[i].item()]))

            if opponent_indices:
                for local_idx in opponent_indices:
                    game = active_games[local_idx]
                    legal = game.get_legal_actions()
                    game.step(int(np.random.choice(legal)))

            new_active = []
            for i in active:
                if not games[i].done:
                    new_active.append(i)
                else:
                    if games[i].winner == model_players[i]:
                        wins += 1
            active = new_active

    model.train()
    win_rate = wins / n_games
    ci_low, ci_high = wilson_confidence(n_games, win_rate)
    return win_rate, ci_low, ci_high


def evaluate_vs_policy(model, opponent_policy, adj_list, n_idioms,
                       max_history_len, max_actions,
                       n_games=1000, device='cuda'):
    """Vectorized model vs arbitrary policy. Batches model inference."""
    model.eval()
    wins = 0

    chunk_size = 256
    for chunk_start in range(0, n_games, chunk_size):
        n_chunk = min(chunk_size, n_games - chunk_start)

        games = []
        for i in range(n_chunk):
            g = IdiomGame(adj_list, n_idioms)
            g.reset()
            games.append(g)
        model_players = [(chunk_start + i) % 2 for i in range(n_chunk)]
        active = list(range(n_chunk))

        while active:
            active_games = [games[i] for i in active]

            model_indices = []
            opponent_indices = []

            for local_idx, orig_idx in enumerate(active):
                g = active_games[local_idx]
                if g.current_player == model_players[orig_idx]:
                    model_indices.append(local_idx)
                else:
                    opponent_indices.append(local_idx)

            if model_indices:
                batch_games = [active_games[i] for i in model_indices]
                batch = prepare_batch_input(batch_games, max_history_len,
                                            max_actions, device)

                with torch.no_grad():
                    action_indices, _, _ = model.get_action(
                        batch['u_ids'], batch['history_ids'],
                        batch['history_mask'],
                        batch['candidate_ids'], batch['candidate_mask'],
                        batch['player_ids'], deterministic=True
                    )

                for i, local_idx in enumerate(model_indices):
                    game = active_games[local_idx]
                    legal = batch['legal_actions'][i]
                    game.step(int(legal[action_indices[i].item()]))

            if opponent_indices:
                for local_idx in opponent_indices:
                    game = active_games[local_idx]
                    legal = game.get_legal_actions()
                    action = opponent_policy(game._state(), legal)
                    game.step(action)

            new_active = []
            for i in active:
                if not games[i].done:
                    new_active.append(i)
                else:
                    if games[i].winner == model_players[i]:
                        wins += 1
            active = new_active

    model.train()
    return wins / n_games


def evaluate_vs_frozen(model, opponent_model, adj_list, n_idioms,
                       max_history_len, max_actions,
                       n_games=400, device='cuda'):
    """Evaluate current model vs frozen opponent. Both play deterministically.

    Returns (win_rate, ci_low, ci_high).
    """
    model.eval()
    opponent_model.eval()
    wins = 0

    chunk_size = 256
    for chunk_start in range(0, n_games, chunk_size):
        n_chunk = min(chunk_size, n_games - chunk_start)

        games = []
        for i in range(n_chunk):
            g = IdiomGame(adj_list, n_idioms)
            g.reset()
            games.append(g)
        model_players = [(chunk_start + i) % 2 for i in range(n_chunk)]
        active = list(range(n_chunk))

        while active:
            active_games = [games[i] for i in active]

            model_indices = []
            frozen_indices = []

            for local_idx, orig_idx in enumerate(active):
                g = active_games[local_idx]
                if g.current_player == model_players[orig_idx]:
                    model_indices.append(local_idx)
                else:
                    frozen_indices.append(local_idx)

            if model_indices:
                batch_games = [active_games[i] for i in model_indices]
                batch = prepare_batch_input(batch_games, max_history_len,
                                            max_actions, device)

                with torch.no_grad():
                    action_indices, _, _ = model.get_action(
                        batch['u_ids'], batch['history_ids'],
                        batch['history_mask'],
                        batch['candidate_ids'], batch['candidate_mask'],
                        batch['player_ids'], deterministic=True
                    )

                for i, local_idx in enumerate(model_indices):
                    game = active_games[local_idx]
                    legal = batch['legal_actions'][i]
                    game.step(int(legal[action_indices[i].item()]))

            if frozen_indices:
                batch_games = [active_games[i] for i in frozen_indices]
                batch = prepare_batch_input(batch_games, max_history_len,
                                            max_actions, device)

                with torch.no_grad():
                    action_indices, _, _ = opponent_model.get_action(
                        batch['u_ids'], batch['history_ids'],
                        batch['history_mask'],
                        batch['candidate_ids'], batch['candidate_mask'],
                        batch['player_ids'], deterministic=True
                    )

                for i, local_idx in enumerate(frozen_indices):
                    game = active_games[local_idx]
                    legal = batch['legal_actions'][i]
                    game.step(int(legal[action_indices[i].item()]))

            new_active = []
            for i in active:
                if not games[i].done:
                    new_active.append(i)
                else:
                    if games[i].winner == model_players[i]:
                        wins += 1
            active = new_active

    model.train()
    win_rate = wins / n_games
    ci_low, ci_high = wilson_confidence(n_games, win_rate)
    return win_rate, ci_low, ci_high


def wilson_confidence(n, p, z=1.96):
    """Wilson confidence interval for a proportion."""
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - spread, center + spread


def export_game_trace(model, opponent, adj_list, n_idioms, idioms,
                      max_history_len, max_actions, model_player=0, device='cuda'):
    """Export a full game trace for human inspection.

    Args:
        model_player: which player (0 or 1) the model controls.
    """
    game = IdiomGame(adj_list, n_idioms)
    game.reset()
    trace = []

    while not game.done:
        legal = game.get_legal_actions()
        if len(legal) == 0:
            break

        is_model = (game.current_player == model_player)

        if is_model:
            inp = prepare_model_input(
                game, max_history_len, max_actions, device
            )
            player_tensor = torch.tensor([game.current_player],
                                         dtype=torch.long, device=device)
            device_type = inp['u_ids'].device.type
            with torch.no_grad(), torch.amp.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=(device_type == 'cuda')):
                logits, value = model(
                    inp['u_ids'], inp['history_ids'], inp['history_mask'],
                    inp['candidate_ids'], inp['candidate_mask'],
                    player_tensor
                )
            logits = logits.float()
            probs = F.softmax(logits, dim=-1).squeeze(0)
            action_idx = probs.argmax().item()
            action = legal[action_idx]

            top_k = min(5, len(legal))
            top_probs, top_idx = probs[:len(legal)].topk(top_k)
            top_actions = [(idioms[legal[i]], f"{p:.3f}")
                           for i, p in zip(top_idx.tolist(), top_probs.tolist())]

            trace.append({
                'step': game.n_steps,
                'player': 'model',
                'current': idioms[game.current],
                'chosen': idioms[action],
                'value': value.item(),
                'top_5': top_actions,
                'n_legal': len(legal),
            })
        else:
            if opponent == 'random':
                action = int(np.random.choice(legal))
            else:
                action = opponent(game._state(), legal)

            trace.append({
                'step': game.n_steps,
                'player': 'opponent',
                'current': idioms[game.current],
                'chosen': idioms[action],
                'n_legal': len(legal),
            })

        game.step(action)

    trace.append({
        'result': f"Player {game.winner} wins",
        'total_steps': game.n_steps,
    })

    return trace
