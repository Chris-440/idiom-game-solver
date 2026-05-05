# Code Review: RL Module

**Reviewer**: Independent code reviewer
**Files reviewed**: src/rl/*.py (10 files)
**Reference**: idiom_rl_plan.md (design document)

---

## File: model.py

### CRITICAL: Missing embedding and encoder variants breaks ablation study

**Lines**: 89-98
**Issue**: `PolicyValueNet.__init__` only supports `embedding_type='char'` and `encoder_type='cross_attention'`. It raises `ValueError` for all other types. The plan defines three embedding variants and three encoder variants for ablation experiments (E1-E6). Without these, experiments E2 (atomic embedding), E3 (mean pool encoder), E4 (no history encoder) cannot be run.
**Fix**: Implement `AtomicIdiomEmbedding`, `MeanPoolEncoder`, and `NoHistoryEncoder` classes (as specified in the plan) and register them in `PolicyValueNet.__init__`.

### CRITICAL: `get_action` uses `@torch.no_grad()` but `action_idx` for masked positions

**Lines**: 124-142
**Issue**: In the `deterministic=False` branch, `torch.multinomial(probs, 1)` is called on the full softmax output including masked positions. While -inf logits produce 0 probability after softmax (so masked positions have 0 probability), if ALL positions are masked (edge case where `len(legal)==0`), softmax produces NaN. This is protected by the caller, but the function has no guard.
**Fix**: Add an explicit guard: if `candidate_mask` has no True values, return a sentinel value or raise. Also consider clamping the mask: `probs = probs.masked_fill(~candidate_mask, 0.0)` before multinomial, then renormalizing.

### MAJOR: No positional encoding in CrossAttentionEncoder

**Lines**: 54-77
**Issue**: The cross-attention encoder has no positional encoding. In a two-player zero-sum game, the current player (whose turn it is) determines the sign of the value function. The model must infer the player from the count of history entries (odd = player 0, even = player 1), but the transformer treats history as a set without position information. While attention weights depend on content, the absolute position count is not directly accessible. This makes it unnecessarily hard for the model to learn the zero-sum value function.
**Fix**: Add a learned position encoding or a binary `current_player` feature to the encoder input. At minimum, add a learned embedding indexed by `len(history) % 2` and add it to the state representation.

### MAJOR: `log_prob` calculation adds `1e-8` to probability before log

**Line**: 139
**Issue**: `torch.log(probs.gather(...) + 1e-8)` adds 1e-8 to the probability before taking the log. This biases the log-probability for high-probability actions (e.g., if true prob=0.9, log(0.9+1e-8) ≈ log(0.9) + 1e-8/0.9). For PPO, the `ratio = exp(new - old)` uses biased log-probs. A cleaner approach uses `F.log_softmax(logits).gather(...)` directly.
**Fix**: Replace with:
```python
log_probs = F.log_softmax(logits, dim=-1)
log_prob = log_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)
```
This avoids the numerical bias entirely. The 1e-8 guard isn't needed because `F.log_softmax` handles numerical stability internally.

### MAJOR: `test_model` only tests one configuration

**Lines**: 147-150
**Issue**: The plan defines 4 model configurations to test, but the implementation only tests `('cross_attention', 'char')`. This means regressions in other encoder/embedding combinations would go undetected.
**Fix**: Restore the full test matrix from the plan:
```python
configs = [
    ('cross_attention', 'char'),
    ('mean_pool', 'char'),
    ('no_history', 'char'),
    ('cross_attention', 'atomic'),
]
```

### MINOR: Inference-time `argmax()` returns index, not action ID

**Lines**: 134-135
**Issue**: `action_idx = probs.argmax(dim=-1)` returns the index into the candidate list, not the idiom ID. This is documented (callers use `legal[action_idx.item()]`), but the function name and docstring don't clarify this. Easy to misuse.
**Fix**: Add a clear docstring stating that the returned index is into `candidate_ids`, not a raw idiom ID.

---

## File: ppo.py

### CRITICAL: `prepare_training_batch` has dead parameters

**Line**: 95
**Issue**: The function signature declares `max_history_len` and `max_actions` parameters, but the function body never uses them. The actual data comes from the saved trajectory tensors. These parameters are misleading and could cause confusion during maintenance.
**Fix**: Remove `max_history_len` and `max_actions` from the function signature.

### CRITICAL: Entropy renormalization changes the policy's entropy

**Lines**: 169-173
**Issue**: The entropy is computed on renormalized probabilities (conditioned on valid actions only), not on the original softmax distribution. The renormalized distribution `valid_probs = probs * mask; valid_probs /= valid_probs.sum()` is the conditional distribution over valid actions. Using its entropy as a bonus means we're encouraging exploration within the valid set, which is the right behavior, but the numerical value differs from the true entropy of the policy. The `1e-8` in the denominator can also produce large values when very few actions are valid.
**Fix**: Document that this is "conditional entropy over valid actions." Consider computing entropy directly from the logits using the log-sum-exp trick to avoid renormalization artifacts.

### MAJOR: Value output is for the current player, but no player indicator in state

**Lines**: - (design issue across model.py and ppo.py)
**Issue**: The value head outputs a single scalar via `Tanh()`, which is trained via MSE against returns computed from the current player's perspective. In self-play, the same network must represent the value for both players. Since the model input has no explicit current-player indicator, the network must infer this from history length parity. This is a fragile inductive leap that slows convergence.
**Fix**: Add the current player as an explicit input feature (e.g., a learned embedding indexed by `current_player` added to the state representation).

### MINOR: Missing `dtype=torch.long` for tensor construction

**Line**: 108
**Issue**: `torch.tensor(all_action_idx, dtype=torch.long, device=device)` is fine since `all_action_idx` is a list of Python ints. But creating individual tensors from Python lists and then moving to device is slightly less efficient than creating them directly on the device.
**Fix**: Minor performance. Not critical.

### MINOR: F.softmax is called twice (in ppo_update and inside the loss)

**Lines**: 152-169
**Issue**: `F.log_softmax(logits, dim=-1)` is called on line 153 for policy loss, and `F.softmax(logits, dim=-1)` is called on line 169 for entropy. These could share computation.
**Fix**: Store `F.softmax(logits, dim=-1)` once and compute `log_probs` via `torch.log(probs + 1e-8)` or `F.log_softmax` once and `probs = torch.exp(log_probs)`.

---

## File: training.py

### CRITICAL: Resume from checkpoint does not restore optimizer or curriculum state

**train_rl.py lines 176-179** (the resume code in train_rl, triggered from training.py)
**Issue**: When resuming from a checkpoint (`--resume PATH`), only `model.state_dict()` is loaded. The optimizer state (Adam momentum, adaptive learning rates), curriculum stage, and iteration counter are discarded. This means gradient statistics are lost and the learning rate scheduler resets to the beginning of the cosine annealing schedule, making the resume essentially start from scratch with pre-initialized weights.
**Fix**: Load and restore optimizer state, curriculum stage, and iteration:
```python
ckpt = torch.load(args.resume, map_location=config.device, weights_only=True)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
curriculum.stage = ckpt.get('curriculum_stage', 1)
# iteration should continue from ckpt['iteration'] + 1
```

### MAJOR: `train()` function lacks input validation for key tensors

**Lines**: 158-167
**Issue**: If `all_trajectories` is empty, training skips the iteration. But if trajectories contain zero steps (game ended immediately), `prepare_training_batch` creates empty tensors, which then crash during PPO update (e.g., `randperm(0)` or empty tensor operations).
**Fix**: Add a check: if the total number of steps across all trajectories is zero, skip the PPO update.

### MAJOR: `collect_rollouts` in training always gives model first-player vs random

**rollout.py lines 99-105** (called from training.py)
**Issue**: When training.py calls `collect_rollouts(model, ..., n_games=1, opponent='random')`, the `game_idx` inside collect_rollouts is always 0 (because only 1 game is collected per call). So `model_player = 0 % 2 = 0`, meaning the model always plays first in random games. This biases the training data toward first-player experience.
**Fix**: Pass `game_idx` as a parameter, or alternate the model's player within the training loop.

### MINOR: Cosine annealing schedule resets each iteration

**Lines**: 97-101
**Issue**: The LR scheduler steps every iteration, but in standard PPO, the LR should either be constant or decay monotonically. The cosine annealing with `T_max=config.max_iterations` and `eta_min=config.lr * 0.1` will decay the LR from 3e-4 to 3e-5 over the full training run. This is fine, but note that it's a monotonic decay (not cyclical) because `T_max = max_iterations`.
**Fix**: No fix needed — this is a valid schedule. Just be aware that early resets restart the decay.

### MINOR: `get_entropy_coef` computed per mini-batch instead of per-iteration

**Lines**: 181-184
**Issue**: `entropy_coef` is recomputed on every mini-batch iteration using the same `iteration` value. This is wasteful but produces the same value since `iteration` doesn't change.
**Fix**: Compute `entropy_coef` once per iteration outside the mini-batch loop.

---

## File: environment.py

### MINOR: `np.isin` performance with large used sets

**Line**: 36
**Issue**: `np.isin(candidates, list(self.used))` has O(|candidates| * |used|) complexity. For long games with hundreds of steps, converting the set to a list and computing isin can be slow.
**Fix**: Use a boolean mask array of size `n_idioms` to track used idioms, giving O(1) lookup per candidate.

### MINOR: `test_environment` Test 1 uses stale `state` variable for assertions

**Lines**: 113-125
**Issue**: The test first calls `state = game.step(1)` (line 113), then `state = game.step(2)` (line 118). After line 118, `state` is the return value of `step(2)`. Then `legal = game.get_legal_actions()` (line 122) uses the game's current state, while `state['done']` and `state['winner']` are from the step(2) return. These should be consistent, but it's confusing that `legal` and `state` come from different calls.
**Fix**: Either check `game.done` directly, or re-fetch state after get_legal_actions.

---

## File: rollout.py

### MINOR: `batch_prepare` is dead code

**Lines**: 68-85
**Issue**: `batch_prepare` is defined but never called anywhere in the codebase. The training pipeline uses individual `prepare_model_input` calls within `collect_rollouts`. This function appears to be leftover from an earlier API.
**Fix**: Either remove it (dead code) or mark it with a comment explaining when it should be used.

### MINOR: `collect_rollouts` doesn't handle max_history_len overflow

**Lines**: 46-49
**Issue**: `prepare_model_input` takes `recent_hist = hist[-max_history_len:]` which correctly truncates. But if the history is shorter than `max_history_len`, the remaining positions are zero-padded. The zero-padding is then embedded as idiom 0 (a real idiom), which wastes computation.
**Fix**: This is a design tradeoff (simplicity vs. efficiency). Acceptable as-is.

### MINOR: Tensor CPU round-trip in Trajectory

**Lines**: 23-26 (rollout.py), 130 (ppo.py)
**Issue**: Trajectory.add_step moves tensors to CPU (`.cpu()`). prepare_training_batch moves them back to device (`.to(device)`). This CPU round-trip is intentional to avoid GPU OOM from accumulated trajectory tensors, but adds overhead.
**Fix**: Consider using pinned memory for faster CPU-GPU transfer if this becomes a bottleneck.

---

## File: evaluation.py

### MAJOR: `export_game_trace` hardcodes model as player 0

**Line**: 107
**Issue**: `is_model = (game.current_player == 0)` always assumes the model plays first. There's no parameter to specify which player the model is. This makes it impossible to generate traces where the model plays second.
**Fix**: Add a `model_player` parameter (default 0) to `export_game_trace`.

### MINOR: `export_game_trace` calls `model(...)` without `@torch.no_grad()`

**Line**: 113
**Issue**: The trace function calls `model(...)` directly (not `model.get_action(...)`). If the model is in training mode, this could accumulate computation graph for the trace. The plan's version uses `forward` inside `get_action` which has `@torch.no_grad()`. The current code doesn't have this guard.
**Fix**: Wrap in `with torch.no_grad():` or call `model.get_action(...)` instead.

### MINOR: `wilson_confidence` is defined but never called

**Lines**: 87-92
**Issue**: The function computes Wilson confidence intervals but is never used in evaluation outputs.
**Fix**: Either call it in `evaluate_vs_random`/`evaluate_vs_policy` to return confidence intervals alongside win rates, or remove it as dead code.

---

## File: data_preparation.py

### MINOR: No validation for empty adjacency

**Line**: 57
**Issue**: If `by_head.get(tail_char, [])` returns an empty list for a tail character that doesn't appear as any head character, the successor list is empty. This is correct for terminal nodes. But there's no explicit logging of how many nodes have no successors.
**Fix**: Add a log line for the number of terminal nodes (zero out-degree) for transparency.

### MINOR: `idiom[:4]` slice is redundant after length filtering

**Line**: 45-46
**Issue**: After filtering to keep only 4-character idioms (line 30), `idiom[:4]` is always a no-op. This is harmless but misleading.
**Fix**: Remove the `[:4]` slice or add a comment explaining it's a safety guard.

---

## File: config.py

### MINOR: `to_dict()` includes private and callable attributes

**Lines**: 65-67
**Issue**: `to_dict()` filters only by `not k.startswith('_')`, which means it includes methods and callable attributes. While `RLConfig` is a plain data class, if future changes add methods, they'd leak into the serialized config. Also includes `idiom_chars`-like attributes that may not be JSON-serializable.
**Fix**: Either use `__dict__` directly or explicitly list serializable fields.

---

## File: train_rl.py

### CRITICAL: Resume does not restore optimizer or curriculum state

**Lines**: 176-179
**Issue**: (Duplicated from training.py notes for completeness.) The resume path only loads `model.state_dict()`. It ignores optimizer, curriculum stage, and iteration counter. Combined with the fact that `train()` creates a fresh optimizer and curriculum, resuming from a checkpoint is effectively:
1. Load model weights
2. Create a new optimizer (no momentum from prior training)
3. Start curriculum from stage 1
4. Start cosine LR schedule from the beginning
This can cause training instability.
**Fix**: Load optimizer, curriculum stage, and iteration from the checkpoint. Pass the iteration to `train()` for continuation.

### MAJOR: `max_history_len` adjustment can invalidate existing model

**Lines**: 61, 150-152
**Issue**: `config.max_history_len = max(config.max_history_len, p99)` silently increases `max_history_len` if data analysis suggests a longer history. If resuming training, the model has been trained with the shorter history and the new longer input won't affect correctness (attention handles variable length), but unexpected changes in input size can mask regressions.
**Fix**: Log a warning when `max_history_len` changes, and hard-code the value after the initial analysis so subsequent resumes are consistent.

### MINOR: Quick test doesn't run unit tests first

**Lines**: 49-98
**Issue**: `--quick` mode skips `run_tests()` and goes directly to data loading and training. If there's an environment or model bug, it will surface during training with harder-to-parse errors.
**Fix**: Call `run_tests()` at the start of `run_quick_test()`.

---

## Cross-cutting / Design Issues

### CRITICAL: Zero-sum perspective handling is fragile

**Files**: ppo.py, model.py
**Issue**: The GAE computation correctly handles zero-sum perspective shifts (negating opponent values). However, the value network has no explicit signal about which player is acting. The entire training pipeline depends on the model inferring the current player from history length parity. This is an implicit signal that could be made explicit with a simple `current_player` embedding added to the state representation. In self-play with a shared value network, this ambiguity slows convergence because the same state representation (without player signal) must encode values of opposite signs.

### MAJOR `train()` function creates optimizer internally

**training.py lines 92-96**
**Issue**: `train()` creates the optimizer from scratch each call. This means resuming training (calling `train()` again with a pre-loaded model) creates a fresh optimizer, losing Adam state. The optimizer should be passed in or created from checkpoint data.
**Fix**: Accept an optional `optimizer` parameter. If provided, use it; if not, create one. Similarly, accept optional `start_iteration` for curriculum state.

### MINOR: Single value head, dual responsibility

**File**: model.py lines 100-105
**Issue**: The single value head with Tanh activation outputs values in [-1, 1]. In self-play, this head must represent values for both players (positive when the current player is winning, negative when losing). Tanh output range [-1, 1] is appropriate for win/loss rewards of +1/-1.

---

## Summary

| Severity | Count | Key Items |
|----------|-------|-----------|
| CRITICAL | 6 | Missing embedding/encoder variants; resume doesn't restore optimizer; dead params in `prepare_training_batch`; missing positional encoding in encoder |
| MAJOR | 6 | No current-player indicator; model always plays first vs random; export hardcodes player 0; all-NaN softmax edge case; entropy renormalization; resume corrupts training |
| MINOR | 12 | Dead code, performance, minor numerical issues, missing validation, code quality |

**Most impactful issue**: The lack of a current-player signal in the model input, combined with the resume checkpoint problem, are the two issues most likely to cause real training problems. The resume bug means any interrupted training loses optimizer state and curriculum progress, which is almost certainly a bug the user will encounter.

**Most subtle issue**: The GAE perspective handling in `compute_gae` is correct but the negation of `gae` when players differ (`gae = delta + gamma * lam * (-gae)`) should be carefully verified against the standard multi-agent PPO formulation. A single sign error here would silently train the policy to maximize the wrong objective.
