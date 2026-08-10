# 成语接龙自博弈训练实验报告

## 目标

验证纯自博弈（Self-Play）能否让神经网络学会成语接龙的最优策略。不引入任何外部教师信号（如 Q-table、人类数据），仅通过模型与自己对弈来进化。

## 实验环境

| 项目 | 配置 |
|------|------|
| 成语库 | 26,108 个有效四字成语 |
| 模型 | PolicyValueNet（CharIdiomEmbedding + CrossAttentionEncoder + ValueHead） |
| 参数量 | 3,996,866 |
| 训练框架 | PPO（Proximal Policy Optimization） |
| 对手基准 | Q-table（Multi-core TD-learning 训练，vs_random=99.85%） |
| 硬件 | NVIDIA GPU + CUDA |

## 版本演进

### V1: 冻结对手自博弈（失败）

**配置:**
- 课程学习三阶段，Stage 3: 70% self + 20% qtable + 10% random
- 自博弈模式：当前模型 vs 冻结历史快照（opponent_model）
- 冻结对手定期更新（vs_frozen > 0.55 时）
- epochs=6, lr=3e-4, clip_eps=0.2
- entropy: 0.02 → 0.001

**结果（200 轮）:**
```
vs_random=0.95  vs_qtable=0.02  entropy=0.012  game_len=6.3
```

**失败原因:** 冻结对手造成"近亲繁殖"——模型学会打败自己的旧版本，但策略坍缩到窄模式。entropy 从 0.08 跌至 0.01，对局仅 6 步即结束。vs_qtable 只有 2%。

---

### V2: 冻结对手 + 高熵（失败）

**配置:**
- 同 V1，但 entropy: 0.05 → 0.005

**结果（1300 轮）:**
```
vs_random=0.998  vs_qtable=0.186  entropy=0.55-0.91  game_len=4.7
```

**失败原因:** 熵提高了但对局更短了，vs_qtable 缓慢爬到 18.6% 后停滞。冻结对手的"窄策略反馈循环"未被打破。

---

### V3: 去掉冻结对手（突破）

**配置:**
- 自博弈模式：**同一模型走双方**（opponent='self', opponent_model=None）
- 双方均随机采样（deterministic=False）
- 每局产生 2 条轨迹（P0+P1 视角）
- epochs=6, lr=3e-4, clip_eps=0.2
- entropy: 0.05 → 0.005
- 纯自博弈，不对抗 Q-table（qtable_ratio=0）

**结果（450 轮）:**
```
Iter   0: vs_random=0.735  vs_qtable=0.003  (随机水平)
Iter 300: vs_random=1.000  vs_qtable=0.701  (首次突破 70%)
Iter 400: vs_random=1.000  vs_qtable=0.757  (峰值)
```

**关键发现:** 去掉冻结对手后，vs_qtable 从 0.3% 飙升至 76%。模型从未训练过对抗 Q-table，纯自博弈自动学会了打败它的策略。

**为什么有效:** 双方都用随机采样的同模型自博弈创造了一个"漏洞自暴露"机制——模型走 P0 时用的窄策略会反噬走 P1 的自己，迫使策略修复弱点。唯一稳定不动点是"无明显漏洞的策略"。

---

### V4: 大 clip + 恒熵（突破瓶颈）

**配置:**
- 同 V3，但 clip_eps: 0.2 → 0.4
- entropy: 0.05 → 0.03（恒定不衰减）
- 从 iter 500 checkpoint 恢复

**结果（500 轮，总 iter 500-1000）:**
```
V3 平台: vs_qtable=0.70-0.76
V4 突破: vs_qtable=0.78-0.80 (iter 850 峰值 0.796)
```

**关键发现:** PPO 的 clip_eps=0.2 在 V3 后期限制了策略跳出局部最优。放宽到 0.4 后突破了 76% 的瓶颈。

---

### V5: On-Policy PPO（epochs=2）

**配置:**
- 同 V4，但 ppo_epochs: 6 → 2, lr: 3e-4 → 1e-3
- entropy: 0.05 → 0.05（恒定）
- 从 iter 1000 checkpoint 恢复

**结果（50 轮，总 iter 1000-1050）:**
```
Iter 1050: vs_qtable=0.814  (首轮即超越 V4 峰值)
```

**关键发现:** 减少轨迹复用次数让训练更"on-policy"，模型从新数据中学习更有效。

---

### V6: 纯 On-Policy（epochs=1）

**配置:**
- 同 V5，但 ppo_epochs: 2 → 1
- 每批轨迹只使用一次即丢弃
- 从 iter 1000 checkpoint 重新开始

**结果（750 轮，总 iter 1000-1750）:**
```
Iter 1050: vs_qtable=0.801
Iter 1150: vs_qtable=0.820
Iter 1650: vs_qtable=0.855  (峰值)
Iter 1750: vs_qtable=0.835
```

**最终: vs_qtable 收敛于 ~84-85%（约 0.849±0.005）**

---

## 最终模型性能（iter 1500 checkpoint）

### 按起始位置分类的先手/后手分析

对 2000 个随机起始成语，模型 vs Q-table 各走一次先手和后手：

| 类型 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 双边都能赢 | 1394 | 69.7% | 模型无论先手后手都赢 |
| 仅模型作为 P0 获胜 | 455 | 22.8% | 当前策略配对下的角色相关结果 |
| 仅模型作为 P1 获胜 | 151 | 7.5% | 当前策略配对下的角色相关结果 |
| 双边都输 | 0 | 0% | 不存在 |

```
Model P0 (先手) 胜率: 0.924
Model P1 (后手) 胜率: 0.772
总体: 0.849
```

### 交叉检查：角色差异是否也出现在 Q-table 自对弈中？

对每类位置回测 Q-table vs Q-table（双方均用 Q 表对弈）：

| 位置类型 | QT-QT P0 赢 | QT-QT P1 赢 | 结论 |
|----------|------------|------------|------|
| Only P0 (455) | **93.8%** | 6.2% | 与 P0 结果高度相关 |
| Only P1 (151) | 8.6% | **91.4%** | 与 P1 结果高度相关 |

**修订结论:** 约 85% 是对当前固定 Q-table 的稳定经验平台期。Q-table 自对弈显示相似的角色相关性，但不能证明这是游戏的数学上限。

```
当前策略对局的经验分解（不是 minimax 上界）:
  双边赢:    1394 位置 × 1.0 = 1394
  仅 P0 获胜: 427 位置 × 0.5 =  213.5
  仅 P1 获胜: 151 位置 × 0.5 =   75.5
  ─────────────────────────────────────
  该计算依赖 PPO 与 Q-table 的现有胜负分类，不能外推至任意策略

尚有 ~28 个位置模型 P1 不如 Q-table P1 (6.2% of 455)
在当前分类假设下的潜在边际改进约为 0.7 个百分点
```

## 各版本对比总表

| 版本 | 核心改动 | vs_qtable 峰值 | 收敛速度 | 结局 |
|------|---------|--------------|---------|------|
| V1 | 冻结对手 | 0.02 | 50轮坍缩 | 失败 |
| V2 | 冻结+高熵 | 0.19 | 1300轮缓爬 | 失败 |
| V3 | 去掉冻结 | 0.76 | 300轮突破 | **转折** |
| V4 | clip_eps=0.4 | 0.80 | 从76→80 | 进步 |
| V5 | epochs=2 | 0.81 | 首轮超越 | 进步 |
| V6 | epochs=1 (纯on-policy) | 0.85 | 750轮收敛 | **最优** |

## 关键结论

### 1. 纯自博弈有效，并在当前基准上形成稳定平台期

不引入任何外部对手（Q-table、人类数据），仅靠同一模型双方对弈，模型从随机水平（0.3% vs Q-table）成长到约 85%。这证明了纯自博弈的经验有效性，但不证明 minimax 最优性。

### 2. 冻结对手是反模式

冻结历史模型作为对手会导致"近亲繁殖"——策略在家族内部循环优化，失去多样性。正确做法是同一模型走双方且保持随机采样。

### 3. On-Policy 优于 Off-Policy

每批轨迹只更新一次（epochs=1）优于多次复用。成语接龙的策略空间离散且 reward 稀疏，用旧策略的数据反复更新会引入偏差。

### 4. 当前基准表现具有显著角色差异

Q-table vs Q-table 自对弈中 P0 胜率为 53.5%，模型 vs Q-table 中 P0 为 92.4%、P1 为 77.2%。约 22% 的采样起点表现为“PPO 仅作为 P0 获胜”；这是策略对局结果，不能直接标记为先手必胜局面。

### 5. 搜索是检验和改进策略的下一步

推理时树搜索（如 MCTS）可用于发现并修正静态策略的局部弱点。其收益应通过多对手稳健性和精确可解子图上的 regret 衡量，而不是表述为“突破理论上限”。

## 最终模型配置

```python
# config.py 参数
lr = 1e-3
clip_eps = 0.4
ppo_epochs = 1
entropy_coef_start = 0.05
entropy_coef_end = 0.05    # 恒定不衰减
n_games_per_iter = 512
max_iterations = 15000
use_frozen_opponent = False  # 同模型双方自博弈
stage1_max_iters = 0         # 跳过课程学习
stage2_max_iters = 0
stage3_self_ratio = 1.0      # 纯自博弈
stage3_random_ratio = 0.0
stage3_qtable_ratio = 0.0
```

## 运行命令

```bash
python src/rl/train_rl.py \
  --stage1-max-iters 0 --stage2-max-iters 0 \
  --stage3-self-ratio 1.0 --stage3-random-ratio 0.0 --stage3-qtable-ratio 0.0 \
  --no-frozen
```
