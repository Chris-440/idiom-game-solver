#!/usr/bin/env python3
"""Interactive play: you vs trained RL model."""

import os
import sys

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rl.config import RLConfig
from src.rl.data_preparation import load_and_index
from src.rl.model import PolicyValueNet
from src.rl.environment import IdiomGame
from src.rl.rollout import prepare_batch_input

def main():
    config = RLConfig()
    print("Loading data + model...")
    data = load_and_index()
    idioms = data['idioms']
    idiom_to_id = data['idiom_to_id']

    # Load latest checkpoint
    ckpt_files = sorted([f for f in os.listdir('checkpoints') if f.endswith('.pt')])
    if not ckpt_files:
        print("No checkpoint found!")
        return
    ckpt_path = os.path.join('checkpoints', ckpt_files[-1])
    print(f"Loading {ckpt_path}")
    device = config.device
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = PolicyValueNet(n_idioms=data['n_idioms'], n_chars=data['n_chars'],
                           idiom_dim=config.idiom_dim, n_heads=config.n_heads,
                           n_layers=config.n_layers)
    model.idiom_emb.set_idiom_chars(data['idiom_chars'])
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    stage = ckpt.get('curriculum_stage', '?')
    iteration = ckpt.get('iteration', '?')
    print(f"Model: iter {iteration}, stage {stage}")
    print(f"Graph: {data['n_idioms']} idioms\n")

    # Choose who goes first
    while True:
        choice = input("你先手还是模型先手? (1=你 / 0=模型): ").strip()
        if choice in ('0', '1'):
            break
    human_player = int(choice)
    model_player = 1 - human_player

    game = IdiomGame(data['adj_list'], data['n_idioms'])
    game.reset()
    print(f"\n开局成语: {idioms[game.current]}")
    print(f"尾字: '{idioms[game.current][-1]}'\n")

    while not game.done:
        legal = game.get_legal_actions()
        if len(legal) == 0:
            break

        if game.current_player == human_player:
            # Your turn
            print(f"--- 你的回合 (当前: {idioms[game.current]}, 尾字: '{idioms[game.current][-1]}') ---")
            print(f"合法走法 ({len(legal)}):")
            # Show first 20 legal moves
            show = legal[:20]
            for i, lid in enumerate(show):
                print(f"  {i}: {idioms[lid]} (尾字: '{idioms[lid][-1]}')")
            if len(legal) > 20:
                print(f"  ... 还有 {len(legal)-20} 个")

            while True:
                try:
                    pick = input("选哪个? (输入编号或成语): ").strip()
                    if pick.isdigit():
                        idx = int(pick)
                        if 0 <= idx < len(show):
                            action = int(show[idx])
                            break
                    else:
                        if pick in idiom_to_id:
                            pid = idiom_to_id[pick]
                            if pid in legal:
                                action = pid
                                break
                    print("无效选择，重试")
                except (EOFError, KeyboardInterrupt):
                    print("\n退出")
                    return

            game.step(action)
            print(f"你选了: {idioms[action]}\n")
        else:
            # Model's turn
            inp = prepare_batch_input([game], config.max_history_len,
                                       config.max_actions, device)
            with torch.no_grad(), torch.amp.autocast(
                    device_type=device,
                    dtype=torch.bfloat16,
                    enabled=(device == 'cuda')):
                logits, value = model(
                    inp['u_ids'], inp['history_ids'], inp['history_mask'],
                    inp['candidate_ids'], inp['candidate_mask'],
                    inp['player_ids'])
            probs = torch.softmax(logits.float(), dim=-1).squeeze(0)

            # Top-3 recommendations
            top_k = min(3, len(legal))
            top_probs, top_idx = probs[:len(legal)].topk(top_k)
            print(f"--- 模型回合 (当前: {idioms[game.current]}, 尾字: '{idioms[game.current][-1]}') ---")
            v = value.item()
            if v > 0.3:    judgment = "乐观"
            elif v > -0.3: judgment = "均势"
            else:          judgment = "悲观"
            print(f"模型判断: {judgment} (value={v:+.3f})")
            print(f"Top-{top_k}:")
            for rank, (i, p) in enumerate(zip(top_idx.tolist(), top_probs.tolist()), 1):
                idiom = idioms[int(legal[i])]
                print(f"  {rank}. {idiom} (尾字: '{idiom[-1]}', prob={p:.3f})")

            # Greedy move
            action_idx = probs[:len(legal)].argmax().item()
            action = int(legal[action_idx])
            game.step(action)
            print(f"模型选了: {idioms[action]}\n")

    # Game over
    print("=" * 40)
    if game.winner == human_player:
        print("你赢了!")
    else:
        print("模型赢了!")
    print(f"总步数: {game.n_steps}")


if __name__ == '__main__':
    main()
