#!/usr/bin/env python3
"""
Interactive checkpoint inspection: query trained model for move recommendations.

Usage:
    python -m src.rl.inspect_model --ckpt checkpoints/ckpt_iter005000.pt
    python -m src.rl.inspect_model --ckpt checkpoints/ckpt_iter005000.pt --top 10
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rl.config import RLConfig
from src.rl.data_preparation import load_and_index
from src.rl.model import PolicyValueNet
from src.rl.environment import IdiomGame
from src.rl.rollout import prepare_model_input


def load_checkpoint(ckpt_path, data, config):
    """Load model from checkpoint."""
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

    ckpt = torch.load(ckpt_path, map_location=config.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    iteration = ckpt.get('iteration', 'unknown')
    stage = ckpt.get('curriculum_stage', 'unknown')
    return model, iteration, stage


def show_recommendations(model, data, config, idiom_str, top_k=5):
    """Show model's top-k recommendations from a given idiom."""
    idioms = data['idioms']
    idiom_to_id = data['idiom_to_id']

    if idiom_str not in idiom_to_id:
        similar = [w for w in idioms if idiom_str[:2] in w][:10]
        print(f"Idiom '{idiom_str}' not found in database.")
        if similar:
            print(f"Similar idioms: {similar}")
        return

    # Set up game state
    game = IdiomGame(data['adj_list'], data['n_idioms'])
    game.reset(start_idiom=idiom_to_id[idiom_str])

    legal = game.get_legal_actions()
    if len(legal) == 0:
        print(f"'{idiom_str}' has no legal moves — it's a dead end!")
        return

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        inp = prepare_model_input(
            game, config.max_history_len, config.max_actions, config.device
        )
        player_tensor = torch.tensor([game.current_player],
                                     dtype=torch.long, device=config.device)
        logits, value = model(
            inp['u_ids'], inp['history_ids'], inp['history_mask'],
            inp['candidate_ids'], inp['candidate_mask'],
            player_tensor
        )
        probs = F.softmax(logits.float(), dim=-1).squeeze(0)

    n_show = min(top_k, len(legal))
    top_probs, top_idx = probs[:len(legal)].topk(n_show)

    print(f"\nCurrent: {idiom_str} (尾字: '{idiom_str[-1]}')")
    print(f"Legal moves: {len(legal)}")
    print(f"Predicted value: {value.item():+.3f}  ({'Winning' if value.item() > 0 else 'Losing'} position)")
    print(f"\nTop-{n_show} recommendations:")
    print(f"{'Rank':<6} {'Idiom':<16} {'Probability':<12} {'Head':<6} {'Tail':<6} {'Degree'}")
    print("-" * 70)
    for rank, (i, p) in enumerate(zip(top_idx.tolist(), top_probs.tolist()), 1):
        idiom = idioms[legal[i]]
        tail = idiom[-1]
        degree = len(data['adj_list'][legal[i]])
        print(f"{rank:<6} {idiom:<16} {p:.4f}       {idiom[0]:<6} {tail:<6} {degree}")


def main():
    parser = argparse.ArgumentParser(description='Inspect trained idiom RL model')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--top', type=int, default=5, help='Number of top recommendations')
    args = parser.parse_args()

    config = RLConfig()
    print(f"Loading data...")
    data = load_and_index(config.idiom_file)

    print(f"Loading checkpoint: {args.ckpt}")
    model, iteration, stage = load_checkpoint(args.ckpt, data, config)
    print(f"Model loaded. Iteration: {iteration}, Stage: {stage}")
    print(f"Device: {config.device}")
    print(f"\nEnter an idiom to see model recommendations.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Idiom > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ('quit', 'exit', 'q'):
            break

        if query:
            show_recommendations(model, data, config, query, args.top)


if __name__ == '__main__':
    main()
