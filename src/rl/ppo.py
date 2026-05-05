import torch
import torch.nn.functional as F
import numpy as np
from .rollout import Trajectory


def compute_gae(trajectory, gamma=0.99, lam=0.95):
    """Standard single-agent GAE.

    The opponent is modelled as part of the environment dynamics.
    trajectory.steps contains only the model's own decision steps.
    Consecutive steps are consecutive MDP transitions for the model agent.

    trajectory.terminal_reward determines the final-step reward:
      +1.0  if the model won
      -1.0  if the model lost
       0.0  if game was truncated (neither — encourages pre-cap wins)
    """
    steps = trajectory.steps
    T = len(steps)

    if T == 0:
        return np.array([]), np.array([])

    advantages = np.zeros(T)
    returns = np.zeros(T)

    gae = 0.0
    for t in reversed(range(T)):
        v_t = steps[t]['value']

        if t == T - 1:
            r_t = trajectory.terminal_reward
            v_next = 0.0
        else:
            r_t = 0.0
            v_next = steps[t + 1]['value']

        delta = r_t + gamma * v_next - v_t
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + v_t

    return advantages, returns


def test_gae():
    """Verify GAE with winner_is_me semantics."""
    # --- Test 1: winner trajectory, gamma=lam=1.0 (MC limit) ---
    traj = Trajectory()
    for t, v in enumerate([0.2, 0.3, 0.5]):
        traj.add_step(
            player=0, u_id=t, action_idx=0, action_id=t + 1,
            log_prob=-0.5, value=v,
            history_ids=torch.zeros(1, 80, dtype=torch.long),
            history_mask=torch.zeros(1, 80, dtype=torch.bool),
            candidate_ids=torch.zeros(1, 600, dtype=torch.long),
            candidate_mask=torch.zeros(1, 600, dtype=torch.bool),
        )
    traj.set_result(True)

    adv, ret = compute_gae(traj, gamma=1.0, lam=1.0)

    # delta[2] = 1 + 0 - 0.5 = 0.5
    # delta[1] = 0 + 0.5 - 0.3 = 0.2
    # delta[0] = 0 + 0.3 - 0.2 = 0.1
    # adv[2] = 0.5
    # adv[1] = 0.2 + 0.5 = 0.7
    # adv[0] = 0.1 + 0.7 = 0.8
    assert abs(adv[2] - 0.5) < 1e-6, f"adv[2] expected 0.5, got {adv[2]}"
    assert abs(adv[1] - 0.7) < 1e-6, f"adv[1] expected 0.7, got {adv[1]}"
    assert abs(adv[0] - 0.8) < 1e-6, f"adv[0] expected 0.8, got {adv[0]}"
    assert abs(ret[0] - 1.0) < 1e-6, f"ret[0] expected 1.0, got {ret[0]}"
    assert adv[0] > 0 and adv[1] > 0 and adv[2] > 0, "winner advantages must be positive"

    # --- Test 2: loser trajectory, gamma=lam=1.0 ---
    traj2 = Trajectory()
    for t, v in enumerate([0.2, 0.3]):
        traj2.add_step(
            player=1, u_id=t, action_idx=0, action_id=t + 1,
            log_prob=-0.5, value=v,
            history_ids=torch.zeros(1, 80, dtype=torch.long),
            history_mask=torch.zeros(1, 80, dtype=torch.bool),
            candidate_ids=torch.zeros(1, 600, dtype=torch.long),
            candidate_mask=torch.zeros(1, 600, dtype=torch.bool),
        )
    traj2.set_result(winner_is_me=False)

    adv2, ret2 = compute_gae(traj2, gamma=1.0, lam=1.0)
    # delta[1] = -1 + 0 - 0.3 = -1.3
    # delta[0] = 0 + 0.3 - 0.2 = 0.1
    # adv[1] = -1.3
    # adv[0] = 0.1 + (-1.3) = -1.2
    assert abs(adv2[1] + 1.3) < 1e-6, f"adv2[1] expected -1.3, got {adv2[1]}"
    assert abs(adv2[0] + 1.2) < 1e-6, f"adv2[0] expected -1.2, got {adv2[0]}"
    assert adv2[0] < 0 and adv2[1] < 0, "loser advantages must be negative"

    # --- Test 3: standard PPO params ---
    adv3, ret3 = compute_gae(traj, gamma=0.99, lam=0.95)
    assert len(adv3) == 3
    assert not np.isnan(adv3).any()
    assert not np.isnan(ret3).any()
    assert adv3[0] > 0 and adv3[1] > 0 and adv3[2] > 0

    # --- Test 4: single-step trajectory ---
    traj4 = Trajectory()
    traj4.add_step(
        player=0, u_id=0, action_idx=0, action_id=1,
        log_prob=-0.5, value=0.5,
        history_ids=torch.zeros(1, 80, dtype=torch.long),
        history_mask=torch.zeros(1, 80, dtype=torch.bool),
        candidate_ids=torch.zeros(1, 600, dtype=torch.long),
        candidate_mask=torch.zeros(1, 600, dtype=torch.bool),
    )
    traj4.set_result(winner_is_me=False)
    adv4, ret4 = compute_gae(traj4, gamma=0.99, lam=0.95)
    # delta = -1 + 0 - 0.5 = -1.5
    assert adv4[0] < 0, f"single-step loser: adv should be negative, got {adv4[0]}"
    assert ret4[0] < 0, f"single-step loser: ret should be negative, got {ret4[0]}"

    print("GAE tests ALL PASSED ✓")


def prepare_training_batch(trajectories, gamma=0.99, lam=0.95, device='cuda'):
    """Convert trajectories to PPO training batch with GAE computation."""
    all_u_ids = []
    all_history_ids = []
    all_history_mask = []
    all_candidate_ids = []
    all_candidate_mask = []
    all_action_idx = []
    all_old_log_probs = []
    all_advantages = []
    all_returns = []
    all_player_ids = []

    for traj in trajectories:
        adv, ret = compute_gae(traj, gamma, lam)

        for t, step in enumerate(traj.steps):
            all_u_ids.append(step['u_id'])
            all_action_idx.append(step['action_idx'])
            all_old_log_probs.append(step['log_prob'])
            all_advantages.append(adv[t])
            all_returns.append(ret[t])
            all_player_ids.append(step['player'])
            all_history_ids.append(step['history_ids'])
            all_history_mask.append(step['history_mask'])
            all_candidate_ids.append(step['candidate_ids'])
            all_candidate_mask.append(step['candidate_mask'])

    # Advantage normalization
    advantages = np.array(all_advantages)
    adv_mean, adv_std = advantages.mean(), advantages.std()
    if adv_std > 1e-8:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    return {
        'u_ids': torch.tensor(all_u_ids, dtype=torch.long, device=device),
        'history_ids': torch.cat(all_history_ids).to(device),
        'history_mask': torch.cat(all_history_mask).to(device),
        'candidate_ids': torch.cat(all_candidate_ids).to(device),
        'candidate_mask': torch.cat(all_candidate_mask).to(device),
        'action_idx': torch.tensor(all_action_idx, dtype=torch.long, device=device),
        'old_log_probs': torch.tensor(all_old_log_probs, dtype=torch.float, device=device),
        'advantages': torch.tensor(advantages, dtype=torch.float, device=device),
        'returns': torch.tensor(all_returns, dtype=torch.float, device=device),
        'player_ids': torch.tensor(all_player_ids, dtype=torch.long, device=device),
    }


def ppo_update(model, batch, optimizer, clip_eps=0.2,
               value_coef=0.5, entropy_coef=0.01, use_amp=True):
    """Single PPO parameter update. Returns metrics dict."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_amp else torch.enable_grad():
        logits, values = model(
            batch['u_ids'],
            batch['history_ids'],
            batch['history_mask'],
            batch['candidate_ids'],
            batch['candidate_mask'],
            batch['player_ids'],
        )
    # logits and values are in bf16; compute loss in fp32 for numerical stability
    logits = logits.float()
    values = values.float()

    # Policy loss (clipped surrogate)
    log_probs = F.log_softmax(logits, dim=-1)
    new_log_probs = log_probs.gather(
        1, batch['action_idx'].unsqueeze(1)
    ).squeeze(1)

    ratio = torch.exp(new_log_probs - batch['old_log_probs'])
    advantages = batch['advantages']

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(values, batch['returns'])

    # Entropy bonus (only on valid actions)
    probs = F.softmax(logits, dim=-1)
    valid_probs = probs * batch['candidate_mask'].float()
    valid_probs = valid_probs / (valid_probs.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(valid_probs * torch.log(valid_probs + 1e-8)).sum(dim=-1)
    entropy = (entropy * (batch['candidate_mask'].any(dim=-1).float())).mean()

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'grad_norm': grad_norm.item(),
        'clip_fraction': ((ratio - 1).abs() > clip_eps).float().mean().item(),
        'approx_kl': (batch['old_log_probs'] - new_log_probs).mean().item(),
    }
