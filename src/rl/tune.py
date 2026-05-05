#!/usr/bin/env python3
"""
Auto-tune RL training parameters for the current hardware.

Measures GPU memory at increasing batch sizes and game counts,
then writes the optimal settings back to config.py.
"""

import sys, os, json, time, gc
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from src.rl.data_preparation import load_and_index, validate_data
from src.rl.model import PolicyValueNet
from src.rl.rollout import collect_rollouts
from src.rl.ppo import prepare_training_batch, ppo_update


def gpu_info():
    if not torch.cuda.is_available():
        return {}
    d = torch.cuda.get_device_properties(0)
    return {
        'name': d.name,
        'total_mb': d.total_memory // (1024 * 1024),
        'free_mb': (d.total_memory - torch.cuda.memory_allocated()) // (1024 * 1024),
        'used_mb': torch.cuda.memory_allocated() // (1024 * 1024),
    }


def benchmark_batch_size(model, data, device, sample_batch, optimizer):
    """Find max batch_size that fits in GPU memory."""
    print("\n=== Batch Size Tuning ===")
    print(f"  GPU: {gpu_info().get('name', 'N/A')}, {gpu_info().get('total_mb', 'N/A')} MB total")

    n_samples = len(sample_batch['u_ids'])
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    results = []

    for bs in sizes:
        if bs > n_samples:
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            t0 = time.time()
            idx = torch.randperm(n_samples, device=device)[:bs]
            mini_batch = {k: v[idx] for k, v in sample_batch.items()
                          if isinstance(v, torch.Tensor)}
            metrics = ppo_update(model, mini_batch, optimizer,
                                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01)
            torch.cuda.synchronize()
            dt = time.time() - t0

            peak_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            print(f"  batch={bs:5d}  |  time={dt*1000:6.1f}ms  |  peak_gpu={peak_mb:5d}MB  |  policy_loss={metrics['policy_loss']:.4f}")

            results.append({'batch_size': bs, 'time_ms': dt * 1000,
                            'peak_gpu_mb': peak_mb, 'ok': True})
            gc.collect()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch={bs:5d}  |  OOM — too large")
                results.append({'batch_size': bs, 'ok': False, 'reason': 'OOM'})
                torch.cuda.empty_cache()
                break
            else:
                raise

    optimal = max((r for r in results if r['ok']), key=lambda r: r['batch_size'],
                  default={'batch_size': 512})
    return optimal['batch_size']


def benchmark_rollout(model, data, config, device):
    """Find optimal n_games for rollout throughput."""
    print("\n=== Rollout Throughput Tuning ===")

    game_counts = [64, 128, 256, 512, 1024, 2048]
    best_throughput = 0
    best_n = 512

    for n in game_counts:
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            trajs = collect_rollouts(model, data['adj_list'], data['n_idioms'],
                                     n_games=n, max_history_len=config.max_history_len,
                                     max_actions=config.max_actions,
                                     opponent='random', device=device)
            torch.cuda.synchronize()
            dt = time.time() - t0
            n_steps = sum(t.length for t in trajs)
            throughput = n_steps / dt
            peak_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            print(f"  games={n:4d}  |  steps={n_steps:4d}  |  time={dt:5.1f}s  |  "
                  f"throughput={throughput:6.0f} steps/s  |  peak_gpu={peak_mb:5d}MB")
            if throughput > best_throughput:
                best_throughput = throughput
                best_n = n
            gc.collect()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  games={n:4d}  |  OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise

    return best_n


def benchmark_all():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"GPU: {gpu_info()}")

    # Load pruned data
    print("\n[1/4] Loading data...")
    data = load_and_index()
    validate_data(data)
    print(f"  Graph: {data['n_idioms']} nodes after pruning")

    # Create model
    print("\n[2/4] Creating model...")
    from src.rl.config import RLConfig
    config = RLConfig()
    model = PolicyValueNet(n_idioms=data['n_idioms'], n_chars=data['n_chars'],
                           idiom_dim=config.idiom_dim, n_heads=config.n_heads,
                           n_layers=config.n_layers)
    model.idiom_emb.set_idiom_chars(data['idiom_chars'])
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Collect sample trajectories for PPO benchmark
    print("\n[3/4] Collecting sample trajectories...")
    trajs = collect_rollouts(model, data['adj_list'], data['n_idioms'],
                             n_games=256, max_history_len=config.max_history_len,
                             max_actions=config.max_actions,
                             opponent='random', device=device)
    n_steps = sum(t.length for t in trajs)
    print(f"  Collected {n_steps} steps from 256 games")

    sample_batch = prepare_training_batch(trajs, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)

    # Tune batch size
    opt_batch_size = benchmark_batch_size(model, data, device, sample_batch, optimizer)
    torch.cuda.empty_cache()

    # Tune rollout size
    opt_n_games = benchmark_rollout(model, data, config, device)
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("RECOMMENDED SETTINGS")
    print("=" * 60)
    print(f"  batch_size       = {opt_batch_size}")
    print(f"  n_games_per_iter = {opt_n_games}")
    print(f"  idiom_dim        = {config.idiom_dim}  (increase if GPU memory remains low)")
    print(f"  n_layers         = {config.n_layers}    (increase for larger model)")

    # How much GPU memory would be used with recommended settings
    gb_used = torch.cuda.max_memory_allocated() / (1024**3)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    print(f"\n  GPU memory peak: {gb_used:.1f} GB / {total_gb:.1f} GB")

    if gb_used < total_gb * 0.5 and torch.cuda.is_available():
        print(f"\n  ⚠️  GPU is underutilized ({gb_used/total_gb*100:.0f}%).")
        print(f"  Consider increasing idiom_dim (256→384 or 512)")
        print(f"  or n_layers (2→4) to build a larger model.")

    return {
        'batch_size': opt_batch_size,
        'n_games_per_iter': opt_n_games,
    }


if __name__ == '__main__':
    results = benchmark_all()
    print(f"\nDone. Optimal config: {json.dumps(results, indent=2)}")
