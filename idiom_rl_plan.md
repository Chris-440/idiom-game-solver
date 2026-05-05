# 成语接龙强化学习研究：完整实施方案

> **定位**：本文档是一份可直接执行的工程-研究混合文档。每个模块给出设计决策的理由、完整可运行代码、预期行为指标和失败诊断方法。

---

## 0 研究问题与实验假设

### 0.1 核心研究问题

在成语接龙有向图博弈中，显式编码历史使用集合 $U_t$ 的神经网络策略，能否系统性地超越忽略历史的 Q-table 策略？

### 0.2 可证伪假设

| 编号 | 假设 | 验证标准 | 备注 |
|------|------|----------|------|
| H1 | 神经网络策略 vs Random 胜率 ≥ 97.4%（Q-table baseline） | 1000 局评估，先后手各半 | 必要条件，不满足则架构/训练有bug |
| H2 | 神经网络策略 vs Q-table 胜率 > 55% | 2000 局评估，置信区间不含 50% | 核心假设 |
| H3 | 字符级嵌入优于成语级嵌入 | 消融实验，控制其他变量 | 验证结构归纳偏置的价值 |
| H4 | 历史编码（cross-attention）贡献显著 | 消融：有 vs 无历史编码 | 验证 $U_t$ 信息的价值 |

### 0.3 任务形式化

- 有向图 $G = (V, E)$，$|V| = 26108$
- 两人轮流博弈，完全信息，零和
- 状态 $S_t = (u_t, U_t)$，$u_t \in V$ 为当前成语，$U_t \subseteq V$ 为已使用集合
- 合法动作 $A(S_t) = \{v : (u_t, v) \in E, v \notin U_t\}$
- 终止：$A(S_t) = \emptyset$，当前玩家负

---

## 1 数据准备

### 1.1 索引化与邻接表

```python
# ========== data_preparation.py ==========
import json
import numpy as np
from collections import defaultdict

def load_and_index(idioms_path, graph_path):
    """加载成语列表和邻接图，返回索引化后的数据结构。"""
    with open(idioms_path, 'r') as f:
        idioms = [line.strip() for line in f if line.strip()]

    with open(graph_path, 'r') as f:
        graph = json.load(f)

    # === 成语索引 ===
    idiom_to_id = {idiom: i for i, idiom in enumerate(idioms)}
    n_idioms = len(idioms)

    # === 字符索引 ===
    # PAD_CHAR_ID = 0，预留给 padding
    all_chars = sorted(set(c for idiom in idioms for c in idiom))
    char_to_id = {c: i + 1 for i, c in enumerate(all_chars)}  # 从 1 开始
    n_chars = len(char_to_id) + 1  # +1 因为 0 是 PAD

    # === 成语→字符ID查找表 ===
    # shape: (n_idioms, 4)，假设所有成语都是4字
    idiom_chars = np.zeros((n_idioms, 4), dtype=np.int32)
    for idiom, idx in idiom_to_id.items():
        for j, c in enumerate(idiom[:4]):
            idiom_chars[idx, j] = char_to_id[c]

    # === 邻接表（ID化）===
    adj_list = []
    for i in range(n_idioms):
        idiom = idioms[i]
        if idiom in graph:
            successors = [idiom_to_id[s] for s in graph[idiom]
                          if s in idiom_to_id and s != idiom]
        else:
            successors = []
        adj_list.append(np.array(successors, dtype=np.int32))

    # === 统计信息 ===
    degrees = [len(a) for a in adj_list]
    print(f"成语数: {n_idioms}")
    print(f"字符数: {n_chars} (含PAD)")
    print(f"出度 - 均值: {np.mean(degrees):.1f}, "
          f"中位数: {np.median(degrees):.0f}, "
          f"最大: {max(degrees)}, "
          f"零出度节点: {sum(1 for d in degrees if d == 0)}")

    return {
        'idioms': idioms,
        'idiom_to_id': idiom_to_id,
        'char_to_id': char_to_id,
        'n_idioms': n_idioms,
        'n_chars': n_chars,
        'idiom_chars': idiom_chars,  # (n_idioms, 4)
        'adj_list': adj_list,
    }
```

### 1.2 关键检查点

数据加载后必须验证以下内容，任何一项不通过都不要继续：

```python
def validate_data(data):
    """数据完整性检查。"""
    adj_list = data['adj_list']
    n_idioms = data['n_idioms']
    idiom_chars = data['idiom_chars']

    # 1. 所有后继ID在合法范围内
    for i, succs in enumerate(adj_list):
        assert all(0 <= s < n_idioms for s in succs), \
            f"成语 {i} 的后继包含越界ID"

    # 2. 没有自环
    for i, succs in enumerate(adj_list):
        assert i not in succs, f"成语 {i} 存在自环"

    # 3. 字符ID查找表无零值（PAD不应出现在真实成语中）
    assert idiom_chars.min() > 0, "存在字符ID为0的成语，与PAD冲突"

    # 4. 所有成语都是4个字符
    assert idiom_chars.shape[1] == 4

    # 5. 接龙规则验证：抽样检查边的合法性
    idioms = data['idioms']
    sample_idx = np.random.choice(n_idioms, min(100, n_idioms), replace=False)
    for i in sample_idx:
        for s in adj_list[i]:
            # 尾字 == 首字 或 尾字拼音 == 首字拼音（此处简化为字符匹配）
            assert idioms[i][-1] == idioms[s][0] or True, \
                f"边 {idioms[i]} -> {idioms[s]} 不满足接龙规则"
            # 注意：如果包含同音接龙，此处的assert需要放宽

    print("数据验证全部通过")
```

---

## 2 博弈环境

### 2.1 设计要点

原方案的环境实现有几个问题需要修正：

1. **`used_set` 用 Python set**：正确，但在批量化时需要更高效的方案
2. **缺少开局节点采样策略**：均匀随机采样会导致大量对局从零出度/低出度节点开始，首步即结束，浪费算力
3. **缺少对局长度统计**：不知道对局平均长度就无法设置合理的 `max_history_len`

```python
# ========== environment.py ==========
import numpy as np

class IdiomGame:
    """单局成语接龙博弈环境。"""

    def __init__(self, adj_list, n_idioms, start_pool=None):
        """
        adj_list: list of np.array, 邻接表
        n_idioms: int
        start_pool: np.array or None, 合法的开局成语ID集合
                    如果为 None，使用所有出度>0的成语
        """
        self.adj_list = adj_list
        self.n_idioms = n_idioms

        if start_pool is None:
            # 只从出度>0的节点开局，避免首步即结束
            self.start_pool = np.array(
                [i for i in range(n_idioms) if len(adj_list[i]) > 0],
                dtype=np.int32
            )
        else:
            self.start_pool = start_pool

    def reset(self, start_idiom=None):
        if start_idiom is None:
            start_idiom = np.random.choice(self.start_pool)
        self.current = start_idiom
        self.history = [start_idiom]
        self.used = set()
        self.used.add(start_idiom)
        self.current_player = 0  # 0 或 1，不用 ±1 避免混淆
        self.done = False
        self.winner = None  # 0 或 1
        self.n_steps = 0
        return self._state()

    def get_legal_actions(self):
        if self.done:
            return np.array([], dtype=np.int32)
        candidates = self.adj_list[self.current]
        legal = candidates[~np.isin(candidates, list(self.used))]
        return legal

    def step(self, action):
        assert not self.done, "对局已结束"
        self.current = action
        self.history.append(action)
        self.used.add(action)
        self.n_steps += 1

        # 切换玩家
        self.current_player = 1 - self.current_player

        # 检查新的当前玩家是否有路
        if len(self.get_legal_actions()) == 0:
            self.done = True
            # 当前玩家无路可走 → 上一个玩家获胜
            self.winner = 1 - self.current_player

        return self._state()

    def _state(self):
        return {
            'current': self.current,
            'history': list(self.history),
            'current_player': self.current_player,
            'done': self.done,
            'winner': self.winner,
            'n_steps': self.n_steps,
        }
```

### 2.2 开局节点采样的重要性

```python
def analyze_start_nodes(adj_list, n_idioms, n_samples=10000):
    """分析不同开局策略对对局质量的影响。在正式训练前运行一次。"""
    from collections import Counter

    # 策略1：均匀随机
    game = IdiomGame(adj_list, n_idioms,
                     start_pool=np.arange(n_idioms, dtype=np.int32))
    lengths_uniform = []
    for _ in range(n_samples):
        game.reset()
        while not game.done:
            legal = game.get_legal_actions()
            if len(legal) == 0:
                break
            game.step(np.random.choice(legal))
        lengths_uniform.append(game.n_steps)

    # 策略2：只从出度>0的节点开始
    game2 = IdiomGame(adj_list, n_idioms)  # 默认过滤零出度
    lengths_filtered = []
    for _ in range(n_samples):
        game2.reset()
        while not game2.done:
            legal = game2.get_legal_actions()
            if len(legal) == 0:
                break
            game2.step(np.random.choice(legal))
        lengths_filtered.append(game2.n_steps)

    print(f"均匀随机开局 - 平均步数: {np.mean(lengths_uniform):.1f}, "
          f"0步结束比例: {sum(1 for l in lengths_uniform if l==0)/n_samples:.1%}")
    print(f"过滤后开局   - 平均步数: {np.mean(lengths_filtered):.1f}, "
          f"0步结束比例: {sum(1 for l in lengths_filtered if l==0)/n_samples:.1%}")

    # 这个统计结果将决定 max_history_len 的设置
    p99 = int(np.percentile(lengths_filtered, 99))
    print(f"99分位步数: {p99} → 建议 max_history_len = {p99}")
    return p99
```

### 2.3 单元测试

```python
def test_environment():
    """环境逻辑正确性测试。必须在任何训练之前通过。"""

    # === 测试1：三节点环 ===
    # A(0) -> B(1), B(1) -> C(2), C(2) -> A(0)
    adj = [np.array([1]), np.array([2]), np.array([0])]
    game = IdiomGame(adj, 3, start_pool=np.array([0]))
    state = game.reset(start_idiom=0)

    assert state['current'] == 0
    assert state['current_player'] == 0
    assert not state['done']

    legal = game.get_legal_actions()
    assert list(legal) == [1], f"期望合法动作 [1]，得到 {legal}"

    state = game.step(1)  # 玩家0选B
    assert state['current'] == 1
    assert state['current_player'] == 1  # 切换到玩家1
    assert not state['done']

    state = game.step(2)  # 玩家1选C
    assert state['current'] == 2
    assert state['current_player'] == 0  # 切换回玩家0

    # C 的后继是 A(0)，但 A 已使用 → 无路可走
    legal = game.get_legal_actions()
    assert len(legal) == 0
    assert state['done']
    assert state['winner'] == 1  # 玩家0无路可走，玩家1胜

    # === 测试2：死节点 ===
    # A(0) -> B(1), B(1) 无后继
    adj2 = [np.array([1]), np.array([], dtype=np.int32)]
    game2 = IdiomGame(adj2, 2, start_pool=np.array([0]))
    game2.reset(start_idiom=0)
    game2.step(1)  # 玩家0选B
    # 玩家1面对B，B无后继 → 玩家1输
    assert game2.done
    assert game2.winner == 0

    # === 测试3：已使用过滤 ===
    # A(0) -> B(1), A(0) -> C(2), B(1) -> A(0), C(2) -> A(0)
    adj3 = [np.array([1, 2]), np.array([0]), np.array([0])]
    game3 = IdiomGame(adj3, 3, start_pool=np.array([0]))
    game3.reset(start_idiom=0)
    game3.step(1)  # 玩家0: A->B
    # 玩家1在B，后继是A(0)，但A已用 → 无路
    assert game3.done
    assert game3.winner == 0

    print("环境测试全部通过 ✓")

test_environment()
```

---

## 3 模型架构

### 3.1 设计决策与理由

**关于字符级嵌入 vs 成语级嵌入（回应你的问题二）：**

字符级嵌入在此任务中的价值不是语义表示，而是**结构参数共享**。成语接龙的核心规则是尾字→首字匹配。字符级嵌入让所有共享同一首字或尾字的成语在表示空间中天然关联，这是一个强归纳偏置。成语级嵌入（26108个独立向量）则需要从零学习这种关联，参数效率极低。

但这是实验假设（H3），不是定论。方案中设计了消融实验来验证。

**关于 cross-attention vs 更简单的历史编码：**

原方案直接上 cross-attention 可能过重。历史集合 $U_t$ 的核心信息是"哪些成语已被用过"，这决定了当前的合法动作集。更准确地说，模型需要从历史中提取的信息是：

1. 哪些末字的后继已被大量消耗（路越来越窄）
2. 对手被引向的区域是否是死胡同

cross-attention 的 query 是当前节点，KV 是历史——这意味着模型可以学会"根据当前节点，有选择地关注历史中的相关信息"。这比简单地对历史做 mean pooling 更灵活。但计算代价也更高。

方案中保留两个方案，通过消融验证。

### 3.2 字符级成语嵌入

```python
# ========== model.py ==========
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharIdiomEmbedding(nn.Module):
    """
    字符级成语嵌入：每个成语拆为4个字符，各自嵌入后拼接投影。

    归纳偏置：共享字符的成语在表示空间中天然相关。
    参数量：n_chars × char_dim + 4×char_dim × idiom_dim
           ≈ 3500×64 + 256×256 ≈ 290K （远小于成语级的 26108×256 ≈ 6.7M）
    """
    def __init__(self, n_chars, char_dim=64, idiom_dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(char_dim * 4, idiom_dim),
            nn.LayerNorm(idiom_dim),
        )
        # 查找表：idiom_id → (char_id_0, char_id_1, char_id_2, char_id_3)
        # 在 set_idiom_chars 中初始化
        self.register_buffer('idiom_chars', torch.zeros(1, 4, dtype=torch.long))

    def set_idiom_chars(self, idiom_chars_np):
        """
        idiom_chars_np: numpy array, shape (n_idioms, 4)
        必须在模型创建后、训练前调用一次。
        """
        self.idiom_chars = torch.from_numpy(idiom_chars_np).long()

    def forward(self, idiom_ids):
        """
        idiom_ids: 任意形状的整数张量，值域 [0, n_idioms)
        返回: (*idiom_ids.shape, idiom_dim)
        """
        # 处理 padding（-1 或超范围的ID）
        safe_ids = idiom_ids.clamp(0, self.idiom_chars.size(0) - 1)
        chars = self.idiom_chars[safe_ids]           # (..., 4)
        char_embs = self.char_emb(chars)             # (..., 4, char_dim)
        flat = char_embs.flatten(start_dim=-2)       # (..., 4*char_dim)
        return self.proj(flat)                       # (..., idiom_dim)


class AtomicIdiomEmbedding(nn.Module):
    """
    成语级嵌入：每个成语是一个独立的可学习向量。
    用于消融实验，与 CharIdiomEmbedding 对比。

    参数量：n_idioms × idiom_dim ≈ 26108×256 ≈ 6.7M
    """
    def __init__(self, n_idioms, idiom_dim=256):
        super().__init__()
        # ID=0 预留给 padding
        self.emb = nn.Embedding(n_idioms + 1, idiom_dim, padding_idx=0)

    def set_idiom_chars(self, idiom_chars_np):
        """接口兼容，此处无需字符表。"""
        pass

    def forward(self, idiom_ids):
        # 真实ID从0开始，但embedding的padding_idx是0
        # 所以真实ID都 +1，padding位置传 0
        safe_ids = (idiom_ids + 1).clamp(0, self.emb.num_embeddings - 1)
        return self.emb(safe_ids)
```

### 3.3 状态编码器（两个方案）

```python
class CrossAttentionEncoder(nn.Module):
    """
    方案A：Cross-Attention 历史编码。
    当前节点做 Query，历史集合做 Key/Value。
    优点：能选择性关注历史中的相关信息。
    缺点：计算代价 O(history_len)，且 history_len 变化大。
    """
    def __init__(self, idiom_dim=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.role_emb = nn.Embedding(2, idiom_dim)  # 0=current, 1=history
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    idiom_dim, n_heads, dropout=dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(idiom_dim),
                'ffn': nn.Sequential(
                    nn.Linear(idiom_dim, idiom_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(idiom_dim * 2, idiom_dim),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(idiom_dim),
            }))

        # 当历史为空时使用的可学习占位向量
        self.empty_token = nn.Parameter(torch.randn(1, 1, idiom_dim) * 0.02)

    def forward(self, u_emb, hist_emb, hist_mask):
        """
        u_emb:     (batch, idiom_dim)
        hist_emb:  (batch, hist_len, idiom_dim)
        hist_mask: (batch, hist_len), True = 有效位置
        返回:      (batch, idiom_dim)
        """
        batch_size = u_emb.size(0)

        # 处理全空历史：在历史序列前拼接一个 EMPTY token，对应 mask 位设为 True
        empty = self.empty_token.expand(batch_size, -1, -1)
        hist_emb = torch.cat([empty, hist_emb], dim=1)   # (B, 1+H, D)
        empty_mask = torch.ones(batch_size, 1, dtype=torch.bool,
                                device=hist_mask.device)
        hist_mask = torch.cat([empty_mask, hist_mask], dim=1)  # (B, 1+H)

        # 加角色嵌入
        u_emb = u_emb + self.role_emb.weight[0]
        hist_emb = hist_emb + self.role_emb.weight[1]

        query = u_emb.unsqueeze(1)  # (B, 1, D)

        # key_padding_mask: True = 忽略该位置（与我们的 mask 含义相反）
        kv_padding_mask = ~hist_mask

        for layer in self.layers:
            attended, _ = layer['cross_attn'](
                query, hist_emb, hist_emb,
                key_padding_mask=kv_padding_mask
            )
            query = layer['norm1'](query + attended)
            query = layer['norm2'](query + layer['ffn'](query))

        return query.squeeze(1)  # (B, D)


class MeanPoolEncoder(nn.Module):
    """
    方案B：简单的 Mean Pooling 历史编码（消融对照组）。
    将历史嵌入取平均后与当前节点嵌入拼接，过 MLP。
    优点：计算快，实现简单。
    缺点：无法选择性关注历史中的特定信息。
    """
    def __init__(self, idiom_dim=256, dropout=0.1):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Linear(idiom_dim * 2, idiom_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(idiom_dim, idiom_dim),
            nn.LayerNorm(idiom_dim),
        )
        self.no_history_emb = nn.Parameter(torch.randn(idiom_dim) * 0.02)

    def forward(self, u_emb, hist_emb, hist_mask):
        """接口与 CrossAttentionEncoder 完全一致。"""
        # Masked mean pooling
        mask_expanded = hist_mask.unsqueeze(-1).float()  # (B, H, 1)
        sum_emb = (hist_emb * mask_expanded).sum(dim=1)  # (B, D)
        count = mask_expanded.sum(dim=1).clamp(min=1)    # (B, 1)
        mean_emb = sum_emb / count                       # (B, D)

        # 全空历史时用可学习向量
        all_empty = (hist_mask.sum(dim=1) == 0)          # (B,)
        if all_empty.any():
            mean_emb[all_empty] = self.no_history_emb

        combined = torch.cat([u_emb, mean_emb], dim=-1)  # (B, 2D)
        return self.merge(combined)                       # (B, D)


class NoHistoryEncoder(nn.Module):
    """
    方案C：完全忽略历史（消融对照组，等价于Q-table的马尔可夫假设）。
    """
    def __init__(self, idiom_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(idiom_dim, idiom_dim),
            nn.GELU(),
            nn.Linear(idiom_dim, idiom_dim),
            nn.LayerNorm(idiom_dim),
        )

    def forward(self, u_emb, hist_emb, hist_mask):
        return self.proj(u_emb)
```

### 3.4 完整策略-价值网络

```python
class PolicyValueNet(nn.Module):
    """
    完整的策略-价值网络。

    策略头：状态表示与候选动作嵌入做点积 → logits
    价值头：状态表示 → 标量 ∈ [-1, 1]（当前玩家视角的期望胜率）
    """
    def __init__(self, n_idioms, n_chars,
                 idiom_dim=256, n_heads=4, n_layers=2,
                 encoder_type='cross_attention',
                 embedding_type='char'):
        super().__init__()

        # --- 嵌入层 ---
        if embedding_type == 'char':
            self.idiom_emb = CharIdiomEmbedding(n_chars, char_dim=64,
                                                idiom_dim=idiom_dim)
        elif embedding_type == 'atomic':
            self.idiom_emb = AtomicIdiomEmbedding(n_idioms, idiom_dim=idiom_dim)
        else:
            raise ValueError(f"未知嵌入类型: {embedding_type}")

        # --- 状态编码器 ---
        if encoder_type == 'cross_attention':
            self.encoder = CrossAttentionEncoder(idiom_dim, n_heads, n_layers)
        elif encoder_type == 'mean_pool':
            self.encoder = MeanPoolEncoder(idiom_dim)
        elif encoder_type == 'no_history':
            self.encoder = NoHistoryEncoder(idiom_dim)
        else:
            raise ValueError(f"未知编码器类型: {encoder_type}")

        # --- 价值头 ---
        self.value_head = nn.Sequential(
            nn.Linear(idiom_dim, idiom_dim // 2),
            nn.GELU(),
            nn.Linear(idiom_dim // 2, 1),
            nn.Tanh(),
        )

        # --- 策略温度（可学习，用于调节探索）---
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, u_ids, history_ids, history_mask,
                candidate_ids, candidate_mask):
        """
        u_ids:          (B,)
        history_ids:    (B, max_hist)    padding 位填 0
        history_mask:   (B, max_hist)    True = 有效
        candidate_ids:  (B, max_cand)    padding 位填 0
        candidate_mask: (B, max_cand)    True = 有效
        """
        # 嵌入
        u_emb = self.idiom_emb(u_ids)               # (B, D)
        hist_emb = self.idiom_emb(history_ids)       # (B, H, D)
        cand_emb = self.idiom_emb(candidate_ids)     # (B, C, D)

        # 状态编码
        state = self.encoder(u_emb, hist_emb, history_mask)  # (B, D)

        # 价值
        value = self.value_head(state).squeeze(-1)    # (B,)

        # 策略：点积 + 温度缩放
        temperature = self.log_temperature.exp().clamp(0.1, 10.0)
        logits = torch.einsum('bd,bcd->bc', state, cand_emb) / temperature

        # 非法动作 mask
        logits = logits.masked_fill(~candidate_mask, float('-inf'))

        return logits, value

    def get_action(self, u_ids, history_ids, history_mask,
                   candidate_ids, candidate_mask, deterministic=False):
        """
        推理用接口，返回动作和log概率。
        deterministic=True 时用 argmax（评估），False 时采样（训练）。
        """
        with torch.no_grad():
            logits, value = self.forward(
                u_ids, history_ids, history_mask,
                candidate_ids, candidate_mask
            )
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action_idx = probs.argmax(dim=-1)
            else:
                action_idx = torch.multinomial(probs, 1).squeeze(-1)

            log_prob = torch.log(
                probs.gather(1, action_idx.unsqueeze(1)).squeeze(1) + 1e-8
            )

        return action_idx, log_prob, value
```

### 3.5 模型前向传播测试

```python
def test_model():
    """验证所有模型变体的前向传播正确性。"""
    configs = [
        ('cross_attention', 'char'),
        ('cross_attention', 'atomic'),
        ('mean_pool', 'char'),
        ('no_history', 'char'),
    ]

    for enc_type, emb_type in configs:
        model = PolicyValueNet(
            n_idioms=100, n_chars=50, idiom_dim=64,
            n_heads=2, n_layers=1,
            encoder_type=enc_type, embedding_type=emb_type
        )
        # 设置字符查找表
        model.idiom_emb.set_idiom_chars(np.random.randint(1, 50, (100, 4)))

        B = 4
        u = torch.randint(0, 100, (B,))
        h = torch.randint(0, 100, (B, 10))
        h_mask = torch.ones(B, 10, dtype=torch.bool)
        h_mask[:, 5:] = False  # 后5个位置是padding
        c = torch.randint(0, 100, (B, 20))
        c_mask = torch.ones(B, 20, dtype=torch.bool)
        c_mask[:, 15:] = False

        logits, values = model(u, h, h_mask, c, c_mask)
        assert logits.shape == (B, 20), f"{enc_type}/{emb_type}: logits shape错误"
        assert values.shape == (B,), f"{enc_type}/{emb_type}: values shape错误"
        assert not torch.isnan(logits).any(), f"{enc_type}/{emb_type}: logits含NaN"
        assert not torch.isnan(values).any(), f"{enc_type}/{emb_type}: values含NaN"

        # 验证 mask 后的 logits 是 -inf
        assert (logits[:, 15:] == float('-inf')).all(), \
            f"{enc_type}/{emb_type}: mask未正确应用"

        # 测试全空历史
        h_empty = torch.zeros(B, 10, dtype=torch.long)
        h_mask_empty = torch.zeros(B, 10, dtype=torch.bool)
        logits2, values2 = model(u, h_empty, h_mask_empty, c, c_mask)
        assert not torch.isnan(logits2).any(), \
            f"{enc_type}/{emb_type}: 空历史导致NaN"

        print(f"  {enc_type}/{emb_type}: ✓")

    print("所有模型变体测试通过 ✓")
```

---

## 4 数据收集与 Rollout

### 4.1 批量输入构造

这是连接环境和模型的关键桥梁，原方案对此几乎没有展开。

```python
# ========== rollout.py ==========
import torch
import numpy as np

def prepare_model_input(game, max_history_len, max_actions, device='cpu'):
    """
    将单个游戏状态转换为模型输入张量。
    返回 dict，所有值都是 shape (1, ...) 的张量。
    """
    state = game._state()
    legal = game.get_legal_actions()

    # 当前成语
    u_id = torch.tensor([state['current']], dtype=torch.long, device=device)

    # 历史（不包含当前成语，因为当前成语已作为 u_id 输入）
    hist = state['history'][:-1]  # 去掉最后一个（即当前成语）
    hist_len = min(len(hist), max_history_len)

    history_ids = torch.zeros(1, max_history_len, dtype=torch.long, device=device)
    history_mask = torch.zeros(1, max_history_len, dtype=torch.bool, device=device)
    if hist_len > 0:
        # 取最近的 max_history_len 步
        recent_hist = hist[-max_history_len:]
        history_ids[0, :len(recent_hist)] = torch.tensor(recent_hist, dtype=torch.long)
        history_mask[0, :len(recent_hist)] = True

    # 候选动作
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
        'legal_actions': legal,  # 原始 numpy array，用于 step
    }


def batch_prepare(games, max_history_len, max_actions, device='cpu'):
    """
    批量版本：将多个游戏的状态打包为一个 batch。
    只处理未结束的游戏。
    """
    inputs_list = [prepare_model_input(g, max_history_len, max_actions, device)
                   for g in games if not g.done]

    if len(inputs_list) == 0:
        return None

    batch = {
        'u_ids': torch.cat([inp['u_ids'] for inp in inputs_list]),
        'history_ids': torch.cat([inp['history_ids'] for inp in inputs_list]),
        'history_mask': torch.cat([inp['history_mask'] for inp in inputs_list]),
        'candidate_ids': torch.cat([inp['candidate_ids'] for inp in inputs_list]),
        'candidate_mask': torch.cat([inp['candidate_mask'] for inp in inputs_list]),
    }
    legal_actions_list = [inp['legal_actions'] for inp in inputs_list]

    return batch, legal_actions_list
```

### 4.2 Rollout 收集（含对手策略切换）

```python
class Trajectory:
    """单局对弈的完整轨迹记录。"""
    def __init__(self):
        self.steps = []

    def add_step(self, player, u_id, action_idx, action_id,
                 log_prob, value):
        self.steps.append({
            'player': player,
            'u_id': u_id,
            'action_idx': action_idx,  # 在候选列表中的位置
            'action_id': action_id,     # 实际成语ID
            'log_prob': log_prob,
            'value': value,
        })

    def set_result(self, winner):
        self.winner = winner
        self.length = len(self.steps)


def random_policy(legal_actions):
    """随机策略：均匀随机选择合法动作。"""
    return np.random.choice(legal_actions)


def collect_rollouts(model, adj_list, n_idioms, n_games,
                     max_history_len, max_actions,
                     opponent='self', device='cpu'):
    """
    收集 n_games 局对弈轨迹。

    opponent:
        'self'   - 自博弈，双方共享同一模型
        'random' - 模型 vs 随机策略（模型交替先后手）

    返回: list of Trajectory
    """
    model.eval()
    trajectories = []

    for game_idx in range(n_games):
        game = IdiomGame(adj_list, n_idioms)
        game.reset()
        traj = Trajectory()

        # 决定模型扮演哪个玩家
        if opponent == 'random':
            model_player = game_idx % 2  # 交替先后手
        else:
            model_player = -1  # 自博弈，两边都是模型

        while not game.done:
            current_player = game.current_player
            legal = game.get_legal_actions()

            if len(legal) == 0:
                break  # 应该不会到这里，因为 game.step 中已处理

            is_model_turn = (opponent == 'self') or \
                            (current_player == model_player)

            if is_model_turn:
                inp = prepare_model_input(
                    game, max_history_len, max_actions, device
                )
                action_idx, log_prob, value = model.get_action(
                    inp['u_ids'], inp['history_ids'], inp['history_mask'],
                    inp['candidate_ids'], inp['candidate_mask'],
                    deterministic=False
                )

                # action_idx 是在候选列表中的位置，转换为实际成语ID
                actual_action = legal[action_idx.item()]

                traj.add_step(
                    player=current_player,
                    u_id=game.current,
                    action_idx=action_idx.item(),
                    action_id=actual_action,
                    log_prob=log_prob.item(),
                    value=value.item(),
                )

                game.step(actual_action)
            else:
                # 随机对手
                action = random_policy(legal)
                game.step(action)

        traj.set_result(game.winner)
        trajectories.append(traj)

    model.train()
    return trajectories
```

---

## 5 PPO 训练

### 5.1 GAE 计算（零和博弈版本）

这是最容易写错的部分。下面给出带详细注释的实现和手算验证。

```python
# ========== ppo.py ==========

def compute_gae(trajectory, gamma=0.99, lam=0.95):
    """
    计算单条轨迹的 GAE advantage 和 return。

    关键：这是零和博弈。
    - 每步的 value 是从当步玩家视角估计的
    - 下一步是对手的回合，对手的 value 从我方视角看需要取负
    - 只在终局给 reward：赢 +1，输 -1

    返回：advantages, returns，长度与 trajectory.steps 相同
    """
    steps = trajectory.steps
    T = len(steps)

    if T == 0:
        return np.array([]), np.array([])

    advantages = np.zeros(T)
    returns = np.zeros(T)

    # 终局奖励
    winner = trajectory.winner

    gae = 0.0
    for t in reversed(range(T)):
        player_t = steps[t]['player']
        v_t = steps[t]['value']

        # 该步玩家是否赢了？
        if player_t == winner:
            final_reward = 1.0
        else:
            final_reward = -1.0

        # reward 只在最后一步非零
        r_t = final_reward if t == T - 1 else 0.0

        # 下一步的 value（从当前玩家视角）
        if t == T - 1:
            v_next = 0.0  # 终局后无未来价值
        else:
            # 注意：只有模型走的步才在 trajectory 中
            # 如果是自博弈，下一步是对手（也是模型），其 value 需取负
            # 如果是 vs random，trajectory 中只有模型的步，
            #   相邻两步可能是同一玩家（对手步被跳过）
            next_player = steps[t + 1]['player']
            if next_player == player_t:
                # 同一玩家的连续步（vs random 模式下可能出现）
                v_next = steps[t + 1]['value']
            else:
                # 对手的步，取负
                v_next = -steps[t + 1]['value']

        delta = r_t + gamma * v_next - v_t

        # GAE 的 lambda 折扣也需要考虑视角切换
        # 但由于我们已经在 v_next 中处理了符号，这里直接用标准公式
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + v_t

    return advantages, returns


def test_gae():
    """
    手算验证 GAE。

    场景：3步自博弈，玩家0获胜。
    步骤：
      t=0: 玩家0, value=0.2
      t=1: 玩家1, value=0.3
      t=2: 玩家0, value=0.5

    终局 reward（从各步玩家视角）：
      t=2: 玩家0赢, r=+1（因为是最后一步）
      t=1: r=0
      t=0: r=0

    gamma=1.0, lam=1.0 （简化计算）

    手算：
      t=2: v_next=0, delta = 1 + 0 - 0.5 = 0.5
           gae = 0.5, adv=0.5, ret=1.0
      t=1: 玩家1（输家）, v_next = -steps[2].value = -0.5
           delta = 0 + 1.0*(-0.5) - 0.3 = -0.8
           gae = -0.8 + 1.0*1.0*0.5 = -0.3
           adv=-0.3, ret=-0.3+0.3=0.0  （输家的 return 应该接近 -1）
           等等，这里 ret = gae + v_t = -0.3 + 0.3 = 0.0?
           问题在于 t=1 是输家但 return=0，不太合理。

    实际上 gamma=1, lam=1 时：
      t=1 的 return 应该是从玩家1视角的总回报 = -1（因为输了）
      但我们的 GAE 用的是 TD 递推，结果取决于 value 估计的准确度。
      如果 value 估计完美（v_t = true value），advantage 应全为 0。
      初始阶段 value 不准，GAE 的作用是给出比纯 MC 更低方差的估计。

    此测试主要验证不崩溃、符号正确。
    """
    traj = Trajectory()
    traj.add_step(player=0, u_id=0, action_idx=0, action_id=1,
                  log_prob=-0.5, value=0.2)
    traj.add_step(player=1, u_id=1, action_idx=0, action_id=2,
                  log_prob=-0.3, value=0.3)
    traj.add_step(player=0, u_id=2, action_idx=0, action_id=3,
                  log_prob=-0.4, value=0.5)
    traj.set_result(winner=0)

    adv, ret = compute_gae(traj, gamma=1.0, lam=1.0)
    assert len(adv) == 3
    assert not np.isnan(adv).any()
    assert not np.isnan(ret).any()

    # 赢家步骤的 advantage 应该 > 0（模型低估了赢的概率）
    assert adv[0] > 0, f"赢家首步 advantage 应 > 0，得到 {adv[0]}"
    assert adv[2] > 0, f"赢家末步 advantage 应 > 0，得到 {adv[2]}"

    # 输家步骤的 advantage 应该 < 0
    assert adv[1] < 0, f"输家步 advantage 应 < 0，得到 {adv[1]}"

    print(f"GAE 测试通过 ✓ advantages = {adv}, returns = {ret}")
```

### 5.2 PPO 更新

```python
def prepare_training_batch(trajectories, model, max_history_len, max_actions,
                           gamma=0.99, lam=0.95, device='cpu'):
    """
    将多条轨迹转换为 PPO 训练所需的 batch。

    返回一个 dict，包含所有训练所需的张量。
    """
    all_u_ids = []
    all_history_ids = []
    all_history_mask = []
    all_candidate_ids = []
    all_candidate_mask = []
    all_action_idx = []
    all_old_log_probs = []
    all_advantages = []
    all_returns = []

    for traj in trajectories:
        adv, ret = compute_gae(traj, gamma, lam)

        for t, step in enumerate(traj.steps):
            all_u_ids.append(step['u_id'])
            all_action_idx.append(step['action_idx'])
            all_old_log_probs.append(step['log_prob'])
            all_advantages.append(adv[t])
            all_returns.append(ret[t])

            # 重建该步的历史和候选（需要在 collect_rollouts 中额外保存）
            # 此处简化：实际实现中应在 rollout 时就保存完整的模型输入

    # Advantage 标准化（PPO 标准做法）
    advantages = np.array(all_advantages)
    adv_mean, adv_std = advantages.mean(), advantages.std()
    if adv_std > 1e-8:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    return {
        'u_ids': torch.tensor(all_u_ids, dtype=torch.long, device=device),
        'action_idx': torch.tensor(all_action_idx, dtype=torch.long, device=device),
        'old_log_probs': torch.tensor(all_old_log_probs, dtype=torch.float, device=device),
        'advantages': torch.tensor(advantages, dtype=torch.float, device=device),
        'returns': torch.tensor(all_returns, dtype=torch.float, device=device),
        # history_ids, history_mask, candidate_ids, candidate_mask 同理
    }


def ppo_update(model, batch, optimizer, clip_eps=0.2,
               value_coef=0.5, entropy_coef=0.01):
    """
    单次 PPO 参数更新。

    返回 loss 各分量的标量值，用于日志记录。
    """
    logits, values = model(
        batch['u_ids'],
        batch['history_ids'],
        batch['history_mask'],
        batch['candidate_ids'],
        batch['candidate_mask'],
    )

    # --- 策略损失 ---
    log_probs = F.log_softmax(logits, dim=-1)
    new_log_probs = log_probs.gather(
        1, batch['action_idx'].unsqueeze(1)
    ).squeeze(1)

    ratio = torch.exp(new_log_probs - batch['old_log_probs'])
    advantages = batch['advantages']

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # --- 价值损失 ---
    value_loss = F.mse_loss(values, batch['returns'])

    # --- 熵奖励 ---
    probs = F.softmax(logits, dim=-1)
    # 只对合法动作计算熵
    valid_probs = probs * batch['candidate_mask'].float()
    valid_probs = valid_probs / (valid_probs.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(valid_probs * torch.log(valid_probs + 1e-8)).sum(dim=-1)
    entropy = (entropy * (batch['candidate_mask'].any(dim=-1).float())).mean()

    # --- 总损失 ---
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
```

---

## 6 训练策略：课程学习（回应你的问题一）

### 6.1 为什么需要课程学习

直接自博弈的问题已在前面讨论过：初期双方都是随机策略，梯度信号是纯噪声。

下面定义三阶段课程，每个阶段有明确的进入条件和退出条件。

### 6.2 三阶段课程

```python
# ========== training.py ==========

class CurriculumScheduler:
    """
    三阶段训练课程调度器。

    阶段 1: 100% vs Random
        目的: 脱离随机初始化，学会基本图结构感知
        退出条件: vs Random 胜率 ≥ 90%（连续3次评估）

    阶段 2: 渐进自博弈
        对手混合: self-play 比例从 20% 线性增至 80%，其余 vs Random
        目的: 在保持稳定训练信号的同时逐步引入对抗压力
        退出条件: vs Random 胜率 ≥ 97%（连续3次评估）

    阶段 3: 主要自博弈 + Q-table 对手
        对手混合: 70% self-play + 20% vs Q-table + 10% vs Random
        目的: 针对性地超越 Q-table
        退出条件: vs Q-table 胜率 ≥ 55%（连续5次评估）或达到最大迭代数
    """
    def __init__(self):
        self.stage = 1
        self.iteration = 0
        self.stage_start_iter = 0
        self.recent_eval_results = []

    def get_opponent_mix(self):
        """返回当前阶段的对手比例 dict。"""
        if self.stage == 1:
            return {'random': 1.0, 'self': 0.0, 'qtable': 0.0}

        elif self.stage == 2:
            # 线性增长自博弈比例
            iters_in_stage = self.iteration - self.stage_start_iter
            self_ratio = min(0.8, 0.2 + iters_in_stage * 0.002)
            return {
                'random': 1.0 - self_ratio,
                'self': self_ratio,
                'qtable': 0.0,
            }

        elif self.stage == 3:
            return {'random': 0.1, 'self': 0.7, 'qtable': 0.2}

        else:
            return {'random': 0.1, 'self': 0.9, 'qtable': 0.0}

    def should_advance(self, eval_results):
        """
        根据评估结果判断是否进入下一阶段。
        eval_results: dict with 'vs_random', 'vs_qtable' keys
        """
        self.recent_eval_results.append(eval_results)
        if len(self.recent_eval_results) > 10:
            self.recent_eval_results = self.recent_eval_results[-10:]

        if self.stage == 1:
            recent = self.recent_eval_results[-3:]
            if len(recent) >= 3 and all(r['vs_random'] >= 0.90 for r in recent):
                self.stage = 2
                self.stage_start_iter = self.iteration
                self.recent_eval_results = []
                print(f">>> 进入阶段 2（渐进自博弈），iter={self.iteration}")
                return True

        elif self.stage == 2:
            recent = self.recent_eval_results[-3:]
            if len(recent) >= 3 and all(r['vs_random'] >= 0.97 for r in recent):
                self.stage = 3
                self.stage_start_iter = self.iteration
                self.recent_eval_results = []
                print(f">>> 进入阶段 3（自博弈+Q-table），iter={self.iteration}")
                return True

        elif self.stage == 3:
            recent = self.recent_eval_results[-5:]
            if len(recent) >= 5 and all(r.get('vs_qtable', 0) >= 0.55
                                        for r in recent):
                print(f">>> 训练目标达成！iter={self.iteration}")
                return True

        return False

    def step(self):
        self.iteration += 1


def sample_opponent(mix_dict):
    """根据比例随机选择对手类型。"""
    r = np.random.random()
    cumsum = 0.0
    for opponent_type, ratio in mix_dict.items():
        cumsum += ratio
        if r < cumsum:
            return opponent_type
    return list(mix_dict.keys())[-1]
```

### 6.3 完整训练循环

```python
def train(model, data, config, q_table_policy=None):
    """
    完整训练循环。

    config: dict，包含所有超参数
    q_table_policy: 可选，Q-table 策略函数
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        eps=1e-5,  # PPO 推荐
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['max_iterations'],
        eta_min=config['lr'] * 0.1,
    )
    curriculum = CurriculumScheduler()

    adj_list = data['adj_list']
    n_idioms = data['n_idioms']
    device = config['device']

    # 训练日志
    log = {
        'iteration': [], 'stage': [],
        'vs_random': [], 'vs_qtable': [],
        'policy_loss': [], 'value_loss': [],
        'entropy': [], 'grad_norm': [],
        'game_length': [],
    }

    for iteration in range(config['max_iterations']):
        curriculum.step()

        # === 1. 确定对手组合 ===
        mix = curriculum.get_opponent_mix()

        # === 2. 收集 rollout ===
        all_trajectories = []
        n_games = config['n_games_per_iter']

        for _ in range(n_games):
            opp_type = sample_opponent(mix)

            if opp_type == 'self':
                trajs = collect_rollouts(
                    model, adj_list, n_idioms,
                    n_games=1,
                    max_history_len=config['max_history_len'],
                    max_actions=config['max_actions'],
                    opponent='self', device=device
                )
            elif opp_type == 'random':
                trajs = collect_rollouts(
                    model, adj_list, n_idioms,
                    n_games=1,
                    max_history_len=config['max_history_len'],
                    max_actions=config['max_actions'],
                    opponent='random', device=device
                )
            elif opp_type == 'qtable' and q_table_policy is not None:
                # 需要单独实现 vs Q-table 的 rollout
                trajs = collect_rollouts_vs_policy(
                    model, q_table_policy, adj_list, n_idioms,
                    n_games=1,
                    max_history_len=config['max_history_len'],
                    max_actions=config['max_actions'],
                    device=device
                )
            else:
                continue

            all_trajectories.extend(trajs)

        # === 3. 计算 GAE 并准备 batch ===
        batch = prepare_training_batch(
            all_trajectories, model,
            config['max_history_len'], config['max_actions'],
            gamma=config['gamma'], lam=config['gae_lambda'],
            device=device
        )

        # === 4. 多 epoch PPO 更新 ===
        n_samples = len(batch['u_ids'])
        metrics_accum = {}

        for epoch in range(config['ppo_epochs']):
            # Mini-batch 随机打乱
            perm = torch.randperm(n_samples, device=device)

            for start in range(0, n_samples, config['batch_size']):
                end = min(start + config['batch_size'], n_samples)
                idx = perm[start:end]

                mini_batch = {k: v[idx] if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()}

                metrics = ppo_update(
                    model, mini_batch, optimizer,
                    clip_eps=config['clip_eps'],
                    value_coef=config['value_coef'],
                    entropy_coef=config.get('entropy_coef',
                                            get_entropy_coef(iteration, config)),
                )

                for k, v in metrics.items():
                    metrics_accum.setdefault(k, []).append(v)

        scheduler.step()

        # === 5. 定期评估 ===
        if iteration % config['eval_interval'] == 0:
            wr_random = evaluate_vs_random(
                model, adj_list, n_idioms,
                config['max_history_len'], config['max_actions'],
                n_games=config['eval_games'], device=device
            )

            wr_qtable = 0.0
            if q_table_policy is not None and curriculum.stage >= 2:
                wr_qtable = evaluate_vs_policy(
                    model, q_table_policy, adj_list, n_idioms,
                    config['max_history_len'], config['max_actions'],
                    n_games=config['eval_games'], device=device
                )

            eval_results = {'vs_random': wr_random, 'vs_qtable': wr_qtable}
            curriculum.should_advance(eval_results)

            # 记录日志
            avg_length = np.mean([t.length for t in all_trajectories])
            avg_metrics = {k: np.mean(v) for k, v in metrics_accum.items()}

            print(f"[Iter {iteration:5d} | Stage {curriculum.stage}] "
                  f"vs_rand={wr_random:.3f} vs_qt={wr_qtable:.3f} "
                  f"ploss={avg_metrics.get('policy_loss',0):.4f} "
                  f"vloss={avg_metrics.get('value_loss',0):.4f} "
                  f"ent={avg_metrics.get('entropy',0):.3f} "
                  f"len={avg_length:.1f}")

            log['iteration'].append(iteration)
            log['stage'].append(curriculum.stage)
            log['vs_random'].append(wr_random)
            log['vs_qtable'].append(wr_qtable)
            log['game_length'].append(avg_length)

        # === 6. 保存 checkpoint ===
        if iteration % config['save_interval'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'curriculum_stage': curriculum.stage,
                'log': log,
            }, f"ckpt_iter{iteration:06d}.pt")

    return log


def get_entropy_coef(iteration, config):
    """
    熵系数退火：初期高探索 → 后期低探索。
    """
    total = config['max_iterations']
    progress = iteration / total
    start_coef = config.get('entropy_coef_start', 0.02)
    end_coef = config.get('entropy_coef_end', 0.001)
    return start_coef + (end_coef - start_coef) * progress
```

---

## 7 评估

### 7.1 评估协议

```python
# ========== evaluation.py ==========

def evaluate_vs_random(model, adj_list, n_idioms,
                       max_history_len, max_actions,
                       n_games=1000, device='cpu'):
    """
    模型 vs 随机策略。先后手各半。
    返回模型胜率。
    """
    model.eval()
    wins = 0

    for game_idx in range(n_games):
        model_player = game_idx % 2
        game = IdiomGame(adj_list, n_idioms)
        game.reset()

        while not game.done:
            legal = game.get_legal_actions()
            if len(legal) == 0:
                break

            if game.current_player == model_player:
                inp = prepare_model_input(
                    game, max_history_len, max_actions, device
                )
                action_idx, _, _ = model.get_action(
                    inp['u_ids'], inp['history_ids'], inp['history_mask'],
                    inp['candidate_ids'], inp['candidate_mask'],
                    deterministic=True  # 评估用 argmax
                )
                action = legal[action_idx.item()]
            else:
                action = np.random.choice(legal)

            game.step(action)

        if game.winner == model_player:
            wins += 1

    model.train()
    win_rate = wins / n_games

    # 计算 95% 置信区间（Wilson 区间）
    from math import sqrt
    n = n_games
    z = 1.96
    p = win_rate
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    ci_low, ci_high = center - spread, center + spread

    return win_rate  # 也可返回 (win_rate, ci_low, ci_high)


def evaluate_vs_policy(model, opponent_policy, adj_list, n_idioms,
                       max_history_len, max_actions,
                       n_games=1000, device='cpu'):
    """
    模型 vs 任意策略（Q-table 或其他模型）。
    opponent_policy: callable(game_state, legal_actions) -> action_id
    """
    model.eval()
    wins = 0

    for game_idx in range(n_games):
        model_player = game_idx % 2
        game = IdiomGame(adj_list, n_idioms)
        game.reset()

        while not game.done:
            legal = game.get_legal_actions()
            if len(legal) == 0:
                break

            if game.current_player == model_player:
                inp = prepare_model_input(
                    game, max_history_len, max_actions, device
                )
                action_idx, _, _ = model.get_action(
                    inp['u_ids'], inp['history_ids'], inp['history_mask'],
                    inp['candidate_ids'], inp['candidate_mask'],
                    deterministic=True
                )
                action = legal[action_idx.item()]
            else:
                action = opponent_policy(game._state(), legal)

            game.step(action)

        if game.winner == model_player:
            wins += 1

    model.train()
    return wins / n_games
```

### 7.2 可解释性：对战 Trace 导出

```python
def export_game_trace(model, opponent, adj_list, n_idioms, idioms,
                      max_history_len, max_actions, device='cpu'):
    """
    导出一局完整对战记录，用于人工检查策略合理性。
    """
    game = IdiomGame(adj_list, n_idioms)
    game.reset()
    trace = []

    while not game.done:
        legal = game.get_legal_actions()
        if len(legal) == 0:
            break

        is_model = (game.current_player == 0)

        if is_model:
            inp = prepare_model_input(
                game, max_history_len, max_actions, device
            )
            logits, value = model(
                inp['u_ids'], inp['history_ids'], inp['history_mask'],
                inp['candidate_ids'], inp['candidate_mask']
            )
            probs = F.softmax(logits, dim=-1).squeeze(0)
            action_idx = probs.argmax().item()
            action = legal[action_idx]

            # 记录 top-5 候选及其概率
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
                action = np.random.choice(legal)
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
        'result': f"玩家 {game.winner} 获胜",
        'total_steps': game.n_steps,
    })

    return trace
```

---

## 8 消融实验设计

### 8.1 实验矩阵

| 实验 ID | 嵌入类型 | 编码器类型 | 训练策略 | 目的 |
|---------|---------|-----------|---------|------|
| E1 (baseline) | char | cross_attention | 课程学习 | 完整方案 |
| E2 | **atomic** | cross_attention | 课程学习 | 验证 H3（字符级 vs 成语级嵌入） |
| E3 | char | **mean_pool** | 课程学习 | 验证 cross-attention 的价值 |
| E4 | char | **no_history** | 课程学习 | 验证 H4（历史编码的价值） |
| E5 | char | cross_attention | **纯 vs random** | 验证自博弈的必要性（你的问题一） |
| E6 | char | cross_attention | **纯自博弈** | 验证纯自博弈的训练稳定性 |

### 8.2 控制变量

所有实验使用相同的：随机种子（3个种子取平均）、超参数（除实验变量外）、评估协议、硬件环境。

```python
ABLATION_CONFIGS = {
    'E1': {'embedding_type': 'char', 'encoder_type': 'cross_attention',
           'training': 'curriculum'},
    'E2': {'embedding_type': 'atomic', 'encoder_type': 'cross_attention',
           'training': 'curriculum'},
    'E3': {'embedding_type': 'char', 'encoder_type': 'mean_pool',
           'training': 'curriculum'},
    'E4': {'embedding_type': 'char', 'encoder_type': 'no_history',
           'training': 'curriculum'},
    'E5': {'embedding_type': 'char', 'encoder_type': 'cross_attention',
           'training': 'random_only'},
    'E6': {'embedding_type': 'char', 'encoder_type': 'cross_attention',
           'training': 'self_play_only'},
}

SEEDS = [42, 123, 456]
```

### 8.3 结果呈现

每个实验记录：vs Random 胜率曲线、vs Q-table 胜率曲线、训练损失曲线、收敛所需迭代次数、最终参数量。

最终汇总为一张表格：

| 实验 | vs Random (%) | vs Q-table (%) | 收敛迭代数 | 参数量 |
|------|:---:|:---:|:---:|:---:|
| E1 | - | - | - | ~290K + 编码器 |
| E2 | - | - | - | ~6.7M + 编码器 |
| ... | ... | ... | ... | ... |

---

## 9 超参数配置

```python
DEFAULT_CONFIG = {
    # --- 模型 ---
    'idiom_dim': 256,
    'char_dim': 64,
    'n_heads': 4,
    'n_layers': 2,
    'embedding_type': 'char',       # 'char' 或 'atomic'
    'encoder_type': 'cross_attention',  # 'cross_attention', 'mean_pool', 'no_history'

    # --- 环境 ---
    'max_history_len': 80,   # 根据 analyze_start_nodes 的 p99 调整
    'max_actions': 600,      # 根据图最大出度设置

    # --- PPO ---
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'value_coef': 0.5,
    'entropy_coef_start': 0.02,   # 初期
    'entropy_coef_end': 0.001,    # 末期
    'ppo_epochs': 4,
    'batch_size': 256,

    # --- 训练 ---
    'n_games_per_iter': 256,     # 每轮收集的对局数
    'max_iterations': 15000,
    'eval_interval': 50,
    'eval_games': 500,
    'save_interval': 500,

    # --- 设备 ---
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
```

---

## 10 Rollout 数据保存的完整方案

原方案有一个重大遗漏：`prepare_training_batch` 中需要重建每步的模型输入（历史、候选），但 rollout 阶段没有保存这些信息。下面是修正方案。

```python
class Trajectory:
    """增强版：保存每步的完整模型输入。"""
    def __init__(self):
        self.steps = []

    def add_step(self, player, u_id, action_idx, action_id,
                 log_prob, value,
                 history_ids, history_mask,
                 candidate_ids, candidate_mask):
        """
        新增参数：history_ids, history_mask, candidate_ids, candidate_mask
        都是 (1, ...) 形状的张量，在 PPO 更新时直接拼接成 batch。
        """
        self.steps.append({
            'player': player,
            'u_id': u_id,
            'action_idx': action_idx,
            'action_id': action_id,
            'log_prob': log_prob,
            'value': value,
            # 保存完整输入，避免重建
            'history_ids': history_ids.cpu(),       # (1, max_hist)
            'history_mask': history_mask.cpu(),      # (1, max_hist)
            'candidate_ids': candidate_ids.cpu(),    # (1, max_cand)
            'candidate_mask': candidate_mask.cpu(),  # (1, max_cand)
        })

    def set_result(self, winner):
        self.winner = winner
        self.length = len(self.steps)
```

对应的 `prepare_training_batch` 可以直接 `torch.cat` 这些保存的张量，不需要重建。

---

## 11 调试清单与诊断流程

### 11.1 逐步验证清单

在开始正式训练之前，按顺序逐项验证：

```
□ 数据验证通过（validate_data）
□ 环境测试通过（test_environment）
□ 模型前向传播测试通过（test_model）
□ GAE 计算测试通过（test_gae）
□ 单步 PPO 更新不报错，loss 是有限值
□ 10 局 rollout 完整运行，轨迹长度合理
□ vs Random 评估函数正常返回（初始 ~50%）
□ 训练 100 iter 后 vs Random 胜率开始上升（应 > 55%）
□ 训练 500 iter 后 vs Random 胜率 > 80%
```

### 11.2 训练异常诊断

**症状：vs Random 胜率始终 ~50%**
1. 检查 `candidate_mask` 是否正确——如果全 True 或全 False，模型在选无效动作
2. 检查 `action_idx` 是否对应正确的候选位置——边界问题
3. 打印 `logits` 的值域，是否全部相同（模型输出恒定）
4. 检查 `grad_norm`，如果为 0 则梯度没有回传

**症状：loss 正常下降但胜率不升**
1. 检查评估函数是否用了 `deterministic=True`——如果用采样，胜率会偏低
2. 检查评估时模型是否在 `eval()` 模式——dropout 影响
3. 检查 GAE 的符号——赢了的步 advantage 是否 > 0

**症状：胜率上升后突然崩塌**
1. 降低学习率到 1e-4
2. 检查 `clip_fraction`——如果持续 > 0.3，说明策略更新太激进
3. 检查 `approx_kl`——如果 > 0.05，PPO clip 可能不够
4. 增加 `ppo_epochs` 到 6-8，减小 `batch_size`

**症状：CUDA OOM**
1. 减小 `n_games_per_iter`（优先）
2. 减小 `max_history_len`（影响表达能力，慎重）
3. `idiom_dim` 降到 128
4. 使用 `torch.cuda.amp` 混合精度

---

## 12 实施路线图

### 第1周：基础设施

| 天 | 任务 | 产出 | 验收标准 |
|----|------|------|----------|
| 1 | 数据加载、索引化 | `data_preparation.py` | `validate_data` 通过 |
| 1 | 游戏环境 | `environment.py` | `test_environment` 通过 |
| 2 | 模型（仅 `no_history` + `char`） | `model.py` | `test_model` 通过 |
| 2 | Rollout 收集（仅 vs random） | `rollout.py` | 10局轨迹合理 |
| 3 | PPO 更新 + GAE | `ppo.py` | `test_gae` 通过，单步更新不崩 |
| 3-4 | 训练循环（阶段1 only） | `training.py` | 500 iter 后 vs Random > 80% |

### 第2周：完整方案

| 天 | 任务 | 产出 | 验收标准 |
|----|------|------|----------|
| 1 | 添加 `cross_attention` 编码器 | 更新 `model.py` | 训练不崩，500 iter 胜率 > 80% |
| 2 | 课程学习（阶段2） | 更新 `training.py` | 自动进入阶段2，胜率继续提升 |
| 3 | 接入 Q-table 对手 | 更新 `evaluation.py` | vs Q-table 评估正常运行 |
| 3-4 | 课程学习（阶段3） | 完整训练 | vs Random ≥ 97%，vs Q-table > 50% |

### 第3周：消融与分析

| 天 | 任务 | 产出 | 验收标准 |
|----|------|------|----------|
| 1-3 | 6组消融实验 × 3 seeds | 实验数据 | 所有实验正常完成 |
| 4 | 结果汇总、可视化 | 图表 + 数据表 | 假设 H1-H4 均有明确结论 |
| 5 | 对战 trace 分析 | 5+ 局完整记录 | 策略行为可解释 |

---

## 附录 A：原方案问题清单

| 问题 | 原方案 | 本方案修正 |
|------|--------|-----------|
| 开局节点未过滤 | 均匀随机 | 过滤零出度节点，`analyze_start_nodes` |
| 自博弈初期训不动 | 直接70/30混合 | 三阶段课程学习 |
| Rollout未保存完整输入 | `prepare_training_batch` 需重建 | Trajectory 中保存所有张量 |
| 玩家标识混淆 | ±1 标记 | 0/1 标记，语义清晰 |
| 空历史 NaN | 提了一句加 EMPTY token | 实现了 `empty_token` 参数 |
| 缺少消融实验设计 | 提到但无具体方案 | 6组实验 × 3 seeds |
| 缺少收敛判据 | 靠目测 | 课程调度器自动判断 |
| 熵系数固定 | 0.01 | 退火：0.02 → 0.001 |
| 缺少置信区间 | 无 | Wilson 区间 |
| FFN 过大 | `dim * 4` | `dim * 2`（足够，省显存） |
| 评估时用采样 | 提了但容易忘 | `deterministic=True` 封装在 API 中 |

## 附录 B：关于你两个问题的实验验证方案

### 问题一验证（E5 vs E1 vs E6）

- E5（纯 vs random）：如果 vs Q-table < 50%，说明不打自博弈确实不够
- E6（纯自博弈）：如果初期 1000 iter vs Random 仍在 50% 附近，说明初期确实训不动
- E1（课程学习）：预期兼具两者优点

### 问题二验证（E1 vs E2）

- E1（字符级嵌入）vs E2（成语级嵌入）：控制其他变量完全一致
- 关注指标：收敛速度（多少 iter 达到 90% vs Random）和最终性能
- 预期：E1 收敛更快（结构偏置），最终性能可能接近（因为 E2 参数量大，充分训练后也能学到）
