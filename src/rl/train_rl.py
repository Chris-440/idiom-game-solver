#!/usr/bin/env python3
"""
Main entry point for idiom solitaire RL training.

Usage:
    python -m src.rl.train_rl                  # Full training
    python -m src.rl.train_rl --test           # Run tests only
    python -m src.rl.train_rl --quick          # Quick small-scale test
    python -m src.rl.train_rl --resume PATH    # Resume from checkpoint
"""

import sys
import os
import argparse
import torch

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rl.config import RLConfig
from src.rl.data_preparation import load_and_index, validate_data
from src.rl.environment import analyze_start_nodes, test_environment
from src.rl.model import PolicyValueNet, test_model
from src.rl.ppo import test_gae
from src.rl.training import train


def run_tests():
    """Run all unit tests before training."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)

    test_environment()
    test_model()
    test_gae()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


def run_quick_test(config):
    """Small-scale training test to verify pipeline."""
    print("=" * 60)
    print("QUICK TEST: Loading data...")
    print("=" * 60)

    data = load_and_index(config.idiom_file)
    validate_data(data)

    # Analyze start nodes
    p99 = analyze_start_nodes(data['adj_list'], data['n_idioms'],
                              n_samples=1000)
    config.max_history_len = max(config.max_history_len, p99)

    print("=" * 60)
    print("QUICK TEST: Creating model...")
    print("=" * 60)

    model = PolicyValueNet(
        n_idioms=data['n_idioms'],
        n_chars=data['n_chars'],
        idiom_dim=config.idiom_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        encoder_type=config.encoder_type,
        embedding_type=config.embedding_type,
    )
    model.idiom_emb.set_idiom_chars(data['idiom_chars'])
    model.to(config.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {config.device}")

    # Quick training test
    test_config = RLConfig(
        max_iterations=50,
        n_games_per_iter=32,
        eval_interval=10,
        eval_games=100,
        save_interval=100,
        max_history_len=config.max_history_len,
        batch_size=64,
        device=config.device,
    )

    print("=" * 60)
    print("QUICK TEST: Starting training...")
    print("=" * 60)

    log = train(model, data, test_config)
    return log


def main():
    parser = argparse.ArgumentParser(description='Idiom Solitaire RL Training')
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests only')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick small-scale training test')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max_iterations')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cpu/cuda/mps)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--entropy-start', type=float, default=None,
                        help='Override entropy_coef_start')
    parser.add_argument('--entropy-end', type=float, default=None,
                        help='Override entropy_coef_end')
    parser.add_argument('--clip-eps', type=float, default=None,
                        help='Override PPO clip_eps')
    parser.add_argument('--stage1-max-iters', type=int, default=None,
                        help='Max iterations in stage 1 (0 to skip)')
    parser.add_argument('--stage2-max-iters', type=int, default=None,
                        help='Max iterations in stage 2 (0 to skip)')
    parser.add_argument('--stage3-self-ratio', type=float, default=None,
                        help='Self-play ratio in stage 3')
    parser.add_argument('--stage3-random-ratio', type=float, default=None,
                        help='Random opponent ratio in stage 3')
    parser.add_argument('--stage3-qtable-ratio', type=float, default=None,
                        help='Q-table opponent ratio in stage 3')
    parser.add_argument('--frozen-update-interval', type=int, default=None,
                        help='Evaluate vs frozen opponent every N iters')
    parser.add_argument('--frozen-win-threshold', type=float, default=None,
                        help='Win rate threshold to update frozen opponent')
    parser.add_argument('--no-frozen', action='store_true',
                        help='Disable frozen opponent (pure same-model self-play)')
    args = parser.parse_args()

    config = RLConfig()

    if args.epochs is not None:
        config.max_iterations = args.epochs
    if args.device is not None:
        config.device = args.device
    if args.lr is not None:
        config.lr = args.lr
    if args.entropy_start is not None:
        config.entropy_coef_start = args.entropy_start
    if args.entropy_end is not None:
        config.entropy_coef_end = args.entropy_end
    if args.clip_eps is not None:
        config.clip_eps = args.clip_eps
    if args.stage1_max_iters is not None:
        config.stage1_max_iters = args.stage1_max_iters
    if args.stage2_max_iters is not None:
        config.stage2_max_iters = args.stage2_max_iters
    if args.stage3_self_ratio is not None:
        config.stage3_self_ratio = args.stage3_self_ratio
    if args.stage3_random_ratio is not None:
        config.stage3_random_ratio = args.stage3_random_ratio
    if args.stage3_qtable_ratio is not None:
        config.stage3_qtable_ratio = args.stage3_qtable_ratio
    if args.frozen_update_interval is not None:
        config.frozen_update_interval = args.frozen_update_interval
    if args.frozen_win_threshold is not None:
        config.frozen_win_threshold = args.frozen_win_threshold
    if args.no_frozen:
        config.use_frozen_opponent = False

    print(f"Device: {config.device}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"LR: {config.lr}")
    print(f"Entropy: {config.entropy_coef_start} -> {config.entropy_coef_end}")
    print(f"Stage 1 max iters: {config.stage1_max_iters}")
    print(f"Stage 2 max iters: {config.stage2_max_iters}")
    print(f"Stage 3 mix: self={config.stage3_self_ratio:.2f} random={config.stage3_random_ratio:.2f} qtable={config.stage3_qtable_ratio:.2f}")
    print(f"Frozen update interval: {config.frozen_update_interval}")
    print(f"Frozen win threshold: {config.frozen_win_threshold}")

    if args.test:
        run_tests()
        return

    if args.quick:
        run_tests()
        run_quick_test(config)
        return

    # Full training
    run_tests()

    print("=" * 60)
    print("FULL TRAINING: Loading data...")
    print("=" * 60)

    data = load_and_index(config.idiom_file)
    validate_data(data)

    p99 = analyze_start_nodes(data['adj_list'], data['n_idioms'],
                              n_samples=1000)
    print(f"  p99={p99}, keeping max_history_len={config.max_history_len}")
    # Do NOT inflate max_history_len. Long games are truncated at the
    # environment step cap and currently receive a neutral terminal reward.

    print("=" * 60)
    print("FULL TRAINING: Creating model...")
    print("=" * 60)

    model = PolicyValueNet(
        n_idioms=data['n_idioms'],
        n_chars=data['n_chars'],
        idiom_dim=config.idiom_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        encoder_type=config.encoder_type,
        embedding_type=config.embedding_type,
    )
    model.idiom_emb.set_idiom_chars(data['idiom_chars'])
    model.to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {config.device}")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        resume_state = {
            'optimizer': ckpt['optimizer'],
            'scheduler': ckpt.get('scheduler'),
            'iteration': ckpt['iteration'],
            'curriculum_stage': ckpt['curriculum_stage'],
            'curriculum_iteration': ckpt.get('curriculum_iteration',
                                              ckpt['iteration'] + 1),
            'curriculum_stage_start_iter': ckpt.get('curriculum_stage_start_iter', 0),
        }
        print(f"Resumed from iteration {ckpt.get('iteration', 'unknown')}")
    else:
        resume_state = None

    print("=" * 60)
    print("FULL TRAINING: Starting...")
    print("=" * 60)

    # Load pre-trained Q-table opponent (baseline: 99.9% vs random)
    print("=" * 60)
    print("FULL TRAINING: Loading Q-table opponent...")
    print("=" * 60)
    import pickle
    qtable_path = os.path.join(config.ckpt_dir, 'qtable.pkl')
    with open(qtable_path, 'rb') as f:
        qt_data = pickle.load(f)
    print(f"Q-table loaded: {qt_data['n_idioms']} nodes, "
          f"vs_random={qt_data.get('vs_random', 'N/A')}")

    def q_table_policy(game_state, legal_actions):
        q = qt_data['q']
        current = game_state['current']
        if len(legal_actions) == 0:
            return -1
        best_a, best_v = int(legal_actions[0]), -1e9
        for a in legal_actions:
            ai = int(a)
            val = q[current].get(ai, 0.0)
            if val > best_v:
                best_v = val
                best_a = ai
        return best_a

    print("=" * 60)
    print("FULL TRAINING: Starting...")
    print("=" * 60)

    log = train(model, data, config,
                q_table_policy=q_table_policy,
                resume_state=resume_state)

    # Save final model
    final_path = os.path.join(config.ckpt_dir, 'final_model.pt')
    torch.save({
        'model': model.state_dict(),
        'log': log,
        'config': config.to_dict(),
    }, final_path)
    print(f"Final model saved to {final_path}")

    # Save training log as JSON
    import json
    log_path = os.path.join(config.results_dir, 'rl_training_log.json')
    with open(log_path, 'w') as f:
        json.dump({k: v if isinstance(v, list) else list(v)
                   for k, v in log.items()}, f, indent=2)
    print(f"Training log saved to {log_path}")


if __name__ == '__main__':
    main()
