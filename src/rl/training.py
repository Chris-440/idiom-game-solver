import torch
import numpy as np
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from .rollout import collect_rollouts, collect_rollouts_vs_policy
from .ppo import prepare_training_batch, ppo_update
from .evaluation import evaluate_vs_random, evaluate_vs_policy, evaluate_vs_frozen, export_game_trace


class CurriculumScheduler:
    """Three-stage curriculum learning scheduler with configurable stage limits."""

    def __init__(self, config):
        self.stage = 1
        self.iteration = 0
        self.stage_start_iter = 0
        self.recent_eval_results = []
        self.stage_max_iters = {
            1: getattr(config, 'stage1_max_iters', 999999),
            2: getattr(config, 'stage2_max_iters', 999999),
        }
        self.stage3_mix = {
            'random': getattr(config, 'stage3_random_ratio', 0.0),
            'self': getattr(config, 'stage3_self_ratio', 1.0),
            'qtable': getattr(config, 'stage3_qtable_ratio', 0.0),
        }
        # Normalize stage3 mix to sum to 1.0
        total = sum(self.stage3_mix.values())
        if total > 0:
            for k in self.stage3_mix:
                self.stage3_mix[k] /= total

    def _force_advance(self):
        """Advance stage(s) if current stage has exceeded max iterations."""
        while self.stage < 3:
            max_iter = self.stage_max_iters.get(self.stage, 999999)
            iters_in_stage = self.iteration - self.stage_start_iter
            if iters_in_stage >= max_iter:
                self.stage += 1
                self.stage_start_iter = self.iteration
                self.recent_eval_results = []
                print(f">>> Stage {self.stage - 1} max iters reached, advancing to Stage {self.stage}, iter={self.iteration}")
            else:
                break

    def get_opponent_mix(self):
        self._force_advance()

        if self.stage == 1:
            return {'random': 1.0, 'self': 0.0, 'qtable': 0.0}

        elif self.stage == 2:
            iters_in_stage = self.iteration - self.stage_start_iter
            self_ratio = min(0.8, 0.2 + iters_in_stage * 0.002)
            return {
                'random': 1.0 - self_ratio,
                'self': self_ratio,
                'qtable': 0.0,
            }

        elif self.stage == 3:
            return dict(self.stage3_mix)

        else:
            return {'random': 0.1, 'self': 0.9, 'qtable': 0.0}

    def should_advance(self, eval_results):
        self.recent_eval_results.append(eval_results)
        if len(self.recent_eval_results) > 10:
            self.recent_eval_results = self.recent_eval_results[-10:]

        if self.stage == 1:
            recent = self.recent_eval_results[-3:]
            if len(recent) >= 3 and all(r['vs_random'] >= 0.90 for r in recent):
                self.stage = 2
                self.stage_start_iter = self.iteration
                self.recent_eval_results = []
                print(f">>> Entering Stage 2 (progressive self-play), iter={self.iteration}")
                return True

        elif self.stage == 2:
            recent = self.recent_eval_results[-3:]
            if len(recent) >= 3 and all(r['vs_random'] >= 0.97 for r in recent):
                self.stage = 3
                self.stage_start_iter = self.iteration
                self.recent_eval_results = []
                print(f">>> Entering Stage 3 (self-play + Q-table), iter={self.iteration}")
                return True

        elif self.stage == 3:
            recent = self.recent_eval_results[-5:]
            if len(recent) >= 5 and all(r.get('vs_qtable', 0) >= 0.55
                                        for r in recent):
                print(f">>> Training objective achieved! iter={self.iteration}")
                return True

        return False

    def step(self):
        self.iteration += 1


def get_entropy_coef(iteration, max_iterations,
                     start_coef=0.02, end_coef=0.001):
    progress = iteration / max_iterations
    return start_coef + (end_coef - start_coef) * progress


def train(model, data, config, q_table_policy=None, resume_state=None):
    """Full training loop with batched vectorized rollouts and curriculum learning."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        eps=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_iterations,
        eta_min=config.lr * 0.1,
    )
    curriculum = CurriculumScheduler(config)

    start_iteration = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        curriculum.stage = resume_state['curriculum_stage']
        curriculum.iteration = resume_state['curriculum_iteration']
        curriculum.stage_start_iter = resume_state.get('curriculum_stage_start_iter', 0)
        start_iteration = resume_state['iteration'] + 1

    adj_list = data['adj_list']
    n_idioms = data['n_idioms']
    device = config.device

    # Create frozen opponent for self-play (optional)
    use_frozen = getattr(config, 'use_frozen_opponent', True)
    opponent_model = None
    if use_frozen:
        opponent_model = copy.deepcopy(model)
        opponent_model.eval()
        for p in opponent_model.parameters():
            p.requires_grad = False

    os.makedirs(config.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(config.tensorboard_dir)

    log = {
        'iteration': [], 'stage': [],
        'vs_random': [], 'vs_qtable': [], 'vs_frozen': [],
        'policy_loss': [], 'value_loss': [],
        'entropy': [], 'grad_norm': [],
        'game_length': [],
    }

    for iteration in range(start_iteration, config.max_iterations):
        curriculum.step()

        mix = curriculum.get_opponent_mix()
        n_games = config.n_games_per_iter

        # ---- Batched rollout by opponent type ----
        all_trajectories = []

        n_self = int(n_games * mix['self'])
        n_random = int(n_games * mix['random'])
        n_qtable = n_games - n_self - n_random

        if n_self > 0:
            trajs = collect_rollouts(
                model, adj_list, n_idioms,
                n_games=n_self,
                max_history_len=config.max_history_len,
                max_actions=config.max_actions,
                max_game_steps=config.max_game_steps,
                opponent='self', device=device,
                opponent_model=opponent_model
            )
            all_trajectories.extend(trajs)

        if n_random > 0:
            trajs = collect_rollouts(
                model, adj_list, n_idioms,
                n_games=n_random,
                max_history_len=config.max_history_len,
                max_actions=config.max_actions,
                max_game_steps=config.max_game_steps,
                opponent='random', device=device
            )
            all_trajectories.extend(trajs)

        if n_qtable > 0 and q_table_policy is not None:
            trajs = collect_rollouts_vs_policy(
                model, q_table_policy, adj_list, n_idioms,
                n_games=n_qtable,
                max_history_len=config.max_history_len,
                max_actions=config.max_actions,
                max_game_steps=config.max_game_steps,
                device=device
            )
            all_trajectories.extend(trajs)

        if len(all_trajectories) == 0:
            continue

        # ---- PPO update ----
        batch = prepare_training_batch(
            all_trajectories,
            gamma=config.gamma, lam=config.gae_lambda,
            device=device
        )

        n_samples = len(batch['u_ids'])
        metrics_accum = {}

        entropy_coef = get_entropy_coef(
            iteration, config.max_iterations,
            config.entropy_coef_start, config.entropy_coef_end
        )

        for epoch in range(config.ppo_epochs):
            perm = torch.randperm(n_samples, device=device)

            for start in range(0, n_samples, config.batch_size):
                end = min(start + config.batch_size, n_samples)
                idx = perm[start:end]

                mini_batch = {k: v[idx] if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()}

                metrics = ppo_update(
                    model, mini_batch, optimizer,
                    clip_eps=config.clip_eps,
                    value_coef=config.value_coef,
                    entropy_coef=entropy_coef,
                    use_amp=getattr(config, 'use_amp', False),
                )

                for k, v in metrics.items():
                    metrics_accum.setdefault(k, []).append(v)

        scheduler.step()

        # ---- Evaluation ----
        if iteration % config.eval_interval == 0:
            wr_random, wr_ci_low, wr_ci_high = evaluate_vs_random(
                model, adj_list, n_idioms,
                config.max_history_len, config.max_actions,
                n_games=config.eval_games, device=device
            )

            wr_qtable = None
            if q_table_policy is not None:  # evaluate whenever opponent is available
                wr_qtable = evaluate_vs_policy(
                    model, q_table_policy, adj_list, n_idioms,
                    config.max_history_len, config.max_actions,
                    n_games=config.eval_games, device=device
                )

            # Frozen opponent evaluation and update
            wr_frozen = None
            if use_frozen and iteration % config.frozen_update_interval == 0:
                wr_frozen, _, _ = evaluate_vs_frozen(
                    model, opponent_model, adj_list, n_idioms,
                    config.max_history_len, config.max_actions,
                    n_games=config.frozen_eval_games, device=device
                )
                if wr_frozen > config.frozen_win_threshold:
                    opponent_model.load_state_dict(model.state_dict())
                    print(f"  [Frozen opponent updated] vs_frozen={wr_frozen:.3f} > {config.frozen_win_threshold}")

            eval_results = {'vs_random': wr_random,
                           'vs_qtable': wr_qtable if wr_qtable is not None else 0.0}
            curriculum.should_advance(eval_results)

            avg_length = np.mean([t.length for t in all_trajectories])
            truncated_ratio = np.mean([float(t.was_truncated) for t in all_trajectories])
            avg_metrics = {k: np.mean(v) for k, v in metrics_accum.items()}

            writer.add_scalar('eval/vs_random', wr_random, iteration)
            if wr_qtable is not None:
                writer.add_scalar('eval/vs_qtable', wr_qtable, iteration)
            if wr_frozen is not None:
                writer.add_scalar('eval/vs_frozen', wr_frozen, iteration)
            writer.add_scalar('eval/game_length', avg_length, iteration)
            writer.add_scalar('eval/truncated_ratio', truncated_ratio, iteration)
            writer.add_scalar('train/policy_loss', avg_metrics['policy_loss'], iteration)
            writer.add_scalar('train/value_loss', avg_metrics['value_loss'], iteration)
            writer.add_scalar('train/entropy', avg_metrics['entropy'], iteration)
            writer.add_scalar('train/grad_norm', avg_metrics['grad_norm'], iteration)
            writer.add_scalar('train/approx_kl', avg_metrics.get('approx_kl', 0), iteration)
            writer.add_scalar('train/clip_fraction', avg_metrics.get('clip_fraction', 0), iteration)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iteration)
            writer.add_scalar('curriculum/stage', curriculum.stage, iteration)

            trace = export_game_trace(
                model, 'random', adj_list, n_idioms, data['idioms'],
                config.max_history_len, config.max_actions,
                model_player=0, device=device
            )
            trace_path = os.path.join(config.log_dir, f'game_trace_iter{iteration:06d}.txt')
            with open(trace_path, 'w') as f:
                for step in trace:
                    f.write(str(step) + '\n')
            writer.add_text('trace/game', str(trace[-1]), iteration)

            qt_str = f"{wr_qtable:.3f}" if wr_qtable is not None else "N/A"
            fr_str = f"{wr_frozen:.3f}" if wr_frozen is not None else "N/A"
            print(f"[Iter {iteration:5d} | Stage {curriculum.stage}] "
                  f"vs_rand={wr_random:.3f} vs_qt={qt_str} vs_frz={fr_str} "
                  f"ploss={avg_metrics.get('policy_loss', 0):.4f} "
                  f"vloss={avg_metrics.get('value_loss', 0):.4f} "
                  f"ent={avg_metrics.get('entropy', 0):.3f} "
                  f"len={avg_length:.1f} trunc={truncated_ratio:.1%}")

            log['iteration'].append(iteration)
            log['stage'].append(curriculum.stage)
            log['vs_random'].append(wr_random)
            log['vs_qtable'].append(wr_qtable if wr_qtable is not None else 0.0)
            log['vs_frozen'].append(wr_frozen)
            log['game_length'].append(avg_length)
            for k, v in avg_metrics.items():
                log.setdefault(k, []).append(v)

        # ---- Checkpoint ----
        if iteration % config.save_interval == 0 and iteration > 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'curriculum_stage': curriculum.stage,
                'curriculum_iteration': curriculum.iteration,
                'curriculum_stage_start_iter': curriculum.stage_start_iter,
                'scheduler': scheduler.state_dict(),
                'log': log,
            }, os.path.join(config.ckpt_dir, f"ckpt_iter{iteration:06d}.pt"))

    writer.close()
    return log
