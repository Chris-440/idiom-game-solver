import torch
import numpy as np
from .environment import IdiomGame


class Trajectory:
    """Stores only the model's own decision steps.

    winner_is_me (bool | None):
        True if the model won, False if lost, None if game was truncated.
    terminal_reward (float):
        ±1 for naturally-terminated games.
        Soft reward in (-0.5, 0.5) for truncated games, based on legal-action ratio.
    """

    def __init__(self):
        self.steps = []
        self.length = 0
        self.winner_is_me = None
        self.terminal_reward = 0.0
        self.was_truncated = False

    def add_step(self, player, u_id, action_idx, action_id,
                 log_prob, value,
                 history_ids, history_mask,
                 candidate_ids, candidate_mask):
        self.steps.append({
            'player': player,
            'u_id': u_id,
            'action_idx': action_idx,
            'action_id': action_id,
            'log_prob': log_prob,
            'value': value,
            'history_ids': history_ids.cpu(),
            'history_mask': history_mask.cpu(),
            'candidate_ids': candidate_ids.cpu(),
            'candidate_mask': candidate_mask.cpu(),
        })

    def set_result(self, winner_is_me):
        self.winner_is_me = winner_is_me
        self.terminal_reward = 1.0 if winner_is_me else -1.0
        self.length = len(self.steps)

    def set_truncated(self):
        """Game hit max_steps — neutral terminal reward, encourages pre-cap wins."""
        self.was_truncated = True
        self.length = len(self.steps)
        self.terminal_reward = 0.0  # draw — model must win before cap


def prepare_model_input(game, max_history_len, max_actions, device='cpu'):
    """Convert single game state to model input tensors (batch_size=1)."""
    state = game._state()
    legal = game.get_legal_actions()

    u_id = torch.tensor([state['current']], dtype=torch.long, device=device)

    hist = state['history'][:-1]
    hist_len = min(len(hist), max_history_len)

    history_ids = torch.zeros(1, max_history_len, dtype=torch.long, device=device)
    history_mask = torch.zeros(1, max_history_len, dtype=torch.bool, device=device)
    if hist_len > 0:
        recent_hist = hist[-max_history_len:]
        history_ids[0, :len(recent_hist)] = torch.tensor(recent_hist, dtype=torch.long)
        history_mask[0, :len(recent_hist)] = True

    n_legal = min(len(legal), max_actions)
    candidate_ids = torch.zeros(1, max_actions, dtype=torch.long, device=device)
    candidate_mask = torch.zeros(1, max_actions, dtype=torch.bool, device=device)
    if n_legal > 0:
        candidate_ids[0, :n_legal] = torch.tensor(legal[:n_legal], dtype=torch.long)
        candidate_mask[0, :n_legal] = True

    return {
        'u_ids': u_id,
        'history_ids': history_ids,
        'history_mask': history_mask,
        'candidate_ids': candidate_ids,
        'candidate_mask': candidate_mask,
        'current_player': state['current_player'],
        'legal_actions': legal,
    }


def prepare_batch_input(games, max_history_len, max_actions, device='cuda'):
    """Prepare batched model input for multiple active games (numpy-accelerated)."""
    n_games = len(games)

    u_ids = torch.zeros(n_games, dtype=torch.long, device=device)
    player_ids = torch.zeros(n_games, dtype=torch.long, device=device)
    history_ids = torch.zeros(n_games, max_history_len, dtype=torch.long, device=device)
    history_mask = torch.zeros(n_games, max_history_len, dtype=torch.bool, device=device)
    candidate_ids = torch.zeros(n_games, max_actions, dtype=torch.long, device=device)
    candidate_mask = torch.zeros(n_games, max_actions, dtype=torch.bool, device=device)

    hist_ids_np = np.zeros((n_games, max_history_len), dtype=np.int64)
    hist_mask_np = np.zeros((n_games, max_history_len), dtype=bool)
    cand_ids_np = np.zeros((n_games, max_actions), dtype=np.int64)
    cand_mask_np = np.zeros((n_games, max_actions), dtype=bool)

    legal_actions = []

    for i, game in enumerate(games):
        state = game._state()
        legal = game.get_legal_actions()
        legal_actions.append(legal)

        u_ids[i] = state['current']
        player_ids[i] = state['current_player']

        hist = state['history'][:-1]
        hist_len = min(len(hist), max_history_len)
        if hist_len > 0:
            recent = hist[-max_history_len:]
            hist_ids_np[i, :len(recent)] = recent
            hist_mask_np[i, :len(recent)] = True

        n_legal = min(len(legal), max_actions)
        if n_legal > 0:
            cand_ids_np[i, :n_legal] = legal[:n_legal]
            cand_mask_np[i, :n_legal] = True

    history_ids.copy_(torch.from_numpy(hist_ids_np))
    history_mask.copy_(torch.from_numpy(hist_mask_np))
    candidate_ids.copy_(torch.from_numpy(cand_ids_np))
    candidate_mask.copy_(torch.from_numpy(cand_mask_np))

    return {
        'u_ids': u_ids,
        'history_ids': history_ids,
        'history_mask': history_mask,
        'candidate_ids': candidate_ids,
        'candidate_mask': candidate_mask,
        'player_ids': player_ids,
        'legal_actions': legal_actions,
    }


def collect_rollouts(model, adj_list, n_idioms, n_games,
                     max_history_len, max_actions, max_game_steps=200,
                     opponent='self', model_player=None, device='cuda',
                     opponent_model=None):
    """Vectorized rollout.

    opponent='self', opponent_model=None:
        Both sides are the current model.
        Produces 2 trajectories per game (P0 + P1 perspectives).

    opponent='self', opponent_model is not None:
        Model vs frozen opponent. Model alternates first/second.
        Frozen opponent plays deterministically.
        Produces 1 trajectory per game (model's perspective only).

    opponent='random':
        Model vs random. model_player alternates first/second.
        Produces 1 trajectory per game (model's perspective).

    max_game_steps: hard game truncation limit.
                    Truncated games get terminal_reward=0 (draw).

    Returns: list of Trajectory.
    """
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()
    max_steps = max_game_steps

    # ---- init all games ----
    games = [IdiomGame(adj_list, n_idioms) for _ in range(n_games)]
    for g in games:
        g.reset()

    frozen_mode = (opponent_model is not None)

    if frozen_mode:
        trajectories = [Trajectory() for _ in range(n_games)]
        if model_player is not None:
            model_players = np.full(n_games, model_player, dtype=np.int32)
        else:
            model_players = np.arange(n_games, dtype=np.int32) % 2
    elif opponent == 'self':
        trajectories = []
        for _ in range(n_games):
            trajectories.append(Trajectory())  # player 0
            trajectories.append(Trajectory())  # player 1
        model_players = np.full(n_games, -1, dtype=np.int32)
    elif model_player is not None:
        trajectories = [Trajectory() for _ in range(n_games)]
        model_players = np.full(n_games, model_player, dtype=np.int32)
    else:
        trajectories = [Trajectory() for _ in range(n_games)]
        model_players = np.arange(n_games, dtype=np.int32) % 2

    active = np.arange(n_games, dtype=np.int32)
    step_count = 0

    while len(active) > 0 and step_count < max_steps:
        active_games = [games[i] for i in active]

        # Separate model-turn vs opponent-turn vs frozen-turn games
        model_indices = []
        frozen_indices = []
        opponent_indices = []

        for local_idx, orig_idx in enumerate(active):
            g = active_games[local_idx]
            cp = g.current_player
            if frozen_mode:
                if cp == model_players[orig_idx]:
                    model_indices.append((local_idx, orig_idx, cp))
                else:
                    frozen_indices.append(local_idx)
            elif opponent == 'self' or cp == model_players[orig_idx]:
                model_indices.append((local_idx, orig_idx, cp))
            else:
                opponent_indices.append(local_idx)

        # ---- Batched model inference (current model, stochastic) ----
        if model_indices:
            batch_games = [active_games[i] for i, _, _ in model_indices]
            batch = prepare_batch_input(batch_games, max_history_len,
                                        max_actions, device)

            with torch.no_grad():
                action_indices, log_probs, values = model.get_action(
                    batch['u_ids'], batch['history_ids'], batch['history_mask'],
                    batch['candidate_ids'], batch['candidate_mask'],
                    batch['player_ids'], deterministic=False
                )

            for i, (local_idx, orig_idx, cp) in enumerate(model_indices):
                game = active_games[local_idx]
                legal = batch['legal_actions'][i]
                act_idx = action_indices[i].item()
                action = int(legal[act_idx])

                if frozen_mode:
                    traj = trajectories[orig_idx]
                elif opponent == 'self':
                    traj = trajectories[orig_idx * 2 + cp]
                else:
                    traj = trajectories[orig_idx]

                traj.add_step(
                    player=cp,
                    u_id=game.current,
                    action_idx=act_idx,
                    action_id=action,
                    log_prob=log_probs[i].item(),
                    value=values[i].item(),
                    history_ids=batch['history_ids'][i:i + 1],
                    history_mask=batch['history_mask'][i:i + 1],
                    candidate_ids=batch['candidate_ids'][i:i + 1],
                    candidate_mask=batch['candidate_mask'][i:i + 1],
                )
                game.step(action)

        # ---- Frozen opponent turns (deterministic) ----
        if frozen_indices:
            batch_games = [active_games[i] for i in frozen_indices]
            batch = prepare_batch_input(batch_games, max_history_len,
                                        max_actions, device)

            with torch.no_grad():
                action_indices, _, _ = opponent_model.get_action(
                    batch['u_ids'], batch['history_ids'], batch['history_mask'],
                    batch['candidate_ids'], batch['candidate_mask'],
                    batch['player_ids'], deterministic=True
                )

            for i, local_idx in enumerate(frozen_indices):
                game = active_games[local_idx]
                legal = batch['legal_actions'][i]
                act_idx = action_indices[i].item()
                action = int(legal[act_idx])
                game.step(action)

        # ---- Opponent turns (random) ----
        if opponent_indices:
            for local_idx in opponent_indices:
                game = active_games[local_idx]
                legal = game.get_legal_actions()
                game.step(int(np.random.choice(legal)))

        # ---- Update active set ----
        still_active = []
        for local_idx, orig_idx in enumerate(active):
            if not active_games[local_idx].done:
                still_active.append(orig_idx)
            else:
                game = active_games[local_idx]
                if frozen_mode:
                    trajectories[orig_idx].set_result(
                        winner_is_me=(game.winner == model_players[orig_idx]))
                elif opponent == 'self':
                    trajectories[orig_idx * 2].set_result(
                        winner_is_me=(game.winner == 0))
                    trajectories[orig_idx * 2 + 1].set_result(
                        winner_is_me=(game.winner == 1))
                else:
                    trajectories[orig_idx].set_result(
                        winner_is_me=(game.winner == model_players[orig_idx]))
        active = np.array(still_active, dtype=np.int32)
        step_count += 1

    # Handle truncated games (hit max_steps before natural termination)
    for i, g in enumerate(games):
        if not g.done:
            if frozen_mode:
                if trajectories[i].length > 0:
                    trajectories[i].set_truncated()
            elif opponent == 'self':
                for ti in [i * 2, i * 2 + 1]:
                    if trajectories[ti].length > 0:
                        trajectories[ti].set_truncated()
            else:
                if trajectories[i].length > 0:
                    trajectories[i].set_truncated()

    # Filter out empty trajectories (player never got a turn)
    trajectories = [t for t in trajectories if t.length > 0]

    model.train()
    return trajectories


def collect_rollouts_vs_policy(model, opponent_policy, adj_list, n_idioms,
                               n_games, max_history_len, max_actions,
                               max_game_steps=200,
                               model_player=None, device='cuda'):
    """Vectorized rollout vs a fixed policy function.

    Produces 1 trajectory per game (model's perspective).
    """
    model.eval()
    max_steps = max_game_steps

    games = [IdiomGame(adj_list, n_idioms) for _ in range(n_games)]
    for g in games:
        g.reset()

    trajectories = [Trajectory() for _ in range(n_games)]

    if model_player is not None:
        model_players = np.full(n_games, model_player, dtype=np.int32)
    else:
        model_players = np.arange(n_games, dtype=np.int32) % 2

    active = np.arange(n_games, dtype=np.int32)
    step_count = 0

    while len(active) > 0 and step_count < max_steps:
        active_games = [games[i] for i in active]

        model_indices = []
        opponent_indices = []

        for local_idx, orig_idx in enumerate(active):
            g = active_games[local_idx]
            if g.current_player == model_players[orig_idx]:
                model_indices.append((local_idx, orig_idx))
            else:
                opponent_indices.append(local_idx)

        if model_indices:
            batch_games = [active_games[i] for i, _ in model_indices]
            batch = prepare_batch_input(batch_games, max_history_len,
                                        max_actions, device)

            with torch.no_grad():
                action_indices, log_probs, values = model.get_action(
                    batch['u_ids'], batch['history_ids'], batch['history_mask'],
                    batch['candidate_ids'], batch['candidate_mask'],
                    batch['player_ids'], deterministic=False
                )

            for i, (local_idx, orig_idx) in enumerate(model_indices):
                game = active_games[local_idx]
                legal = batch['legal_actions'][i]
                act_idx = action_indices[i].item()
                action = int(legal[act_idx])

                trajectories[orig_idx].add_step(
                    player=game.current_player,
                    u_id=game.current,
                    action_idx=act_idx,
                    action_id=action,
                    log_prob=log_probs[i].item(),
                    value=values[i].item(),
                    history_ids=batch['history_ids'][i:i + 1],
                    history_mask=batch['history_mask'][i:i + 1],
                    candidate_ids=batch['candidate_ids'][i:i + 1],
                    candidate_mask=batch['candidate_mask'][i:i + 1],
                )
                game.step(action)

        if opponent_indices:
            for local_idx in opponent_indices:
                game = active_games[local_idx]
                legal = game.get_legal_actions()
                action = opponent_policy(game._state(), legal)
                game.step(action)

        still_active = []
        for local_idx, orig_idx in enumerate(active):
            if not active_games[local_idx].done:
                still_active.append(orig_idx)
            else:
                trajectories[orig_idx].set_result(
                    winner_is_me=(active_games[local_idx].winner == model_players[orig_idx]))
        active = np.array(still_active, dtype=np.int32)
        step_count += 1

    # Handle truncated games
    for i, g in enumerate(games):
        if not g.done and trajectories[i].length > 0:
            trajectories[i].set_truncated()

    trajectories = [t for t in trajectories if t.length > 0]
    model.train()
    return trajectories
