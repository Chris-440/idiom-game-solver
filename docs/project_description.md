# 成语接龙博弈论研究项目 - 详细技术描述

## 项目概述

本项目将成语接龙游戏形式化为有向图上的公平博弈（Impartial Game），应用 Sprague-Grundy 定理分析必胜/必败态，并设计多种求解算法。由于状态空间规模巨大（26,108 个节点），精确求解面临指数级复杂度瓶颈，因此主要采用基于强化学习的近似求解方法。

---

## 一、问题定义与数学建模

### 1.1 游戏规则形式化

成语接龙被建模为有向图 $G = (V, E)$：
- **节点集 $V$**：成语库中的所有四字成语
- **边集 $E$**：若成语 $u$ 的尾字与成语 $v$ 的首字相同（或拼音相同），则存在有向边 $u \to v$

**状态定义**：$S_t = (u_t, U_t)$
- $u_t \in V$：当前接龙的最后一个成语
- $U_t \subseteq V$：已使用的成语集合

**合法动作空间**：
$$A(S_t) = \{ v \mid (u_t, v) \in E \land v \notin U_t \}$$

**胜负判定**：若 $A(S_t) = \emptyset$，当前玩家判负。

### 1.2 数据规模与预处理

**原始数据**：
- 成语数量：29,502 个
- 边数量：713,426 条
- 平均出度：约 24.2

**死端节点检测与迭代剪枝**：
严格同字接龙图中有 3,183 个初始出度为 0 的死端节点；使用迭代剪枝持续移除新产生的死端：
```
while 存在出度为0的节点:
    移除该节点
    更新邻居节点的出度
```
**结果**：移除 3,394 个迭代死端节点，保留 26,108 个节点。

**预处理代码（Go 语言）**：
```go
func PruneDeadEnds(g *Graph) *Graph {
    valid := make([]bool, g.N)
    for i := range valid { valid[i] = true }
    
    for {
        changed := false
        for u := 0; u < g.N; u++ {
            if !valid[u] { continue }
            outDeg := 0
            for _, v := range g.Adj[u] {
                if valid[v] { outDeg++ }
            }
            if outDeg == 0 {
                valid[u] = false
                changed = true
            }
        }
        if !changed { break }
    }
    // 重建压缩后的图...
}
```

---

## 二、核心代码架构

### 2.1 项目结构

```
成语接龙/
├── src/                      # Python 核心模块
│   ├── idiom_data.py         # 成语数据加载与索引
│   ├── idiom_graph.py        # 有向图数据结构
│   ├── sg_solver.py          # Sprague-Grundy 求解器
│   ├── minimax_solver.py     # Minimax + Alpha-Beta 求解器
│   ├── q_solver.py           # Value Iteration 求解器
│   ├── selfplay_solver.py    # Q-Learning 自对抗求解器
│   ├── experiment.py         # 实验框架
│   └── methods/              # 多方法对比模块
│       ├── value_iteration_player.py
│       ├── q_learning_player.py
│       └── mcts_player.py
├── go/                       # Go 语言高性能实现
│   ├── idiom.go              # 图构建与剪枝
│   ├── train_dense.go        # 密集采样训练（500万局）
│   └── check_games.go        # 对战验证
├── tests/
│   └── test_solver.py        # 单元测试
├── data/
│   ├── idiom.json            # 成语数据源
│   └── chinese-xinhua-master/ # 新华词典数据集
└──── results/
    ├── training_curve_dense.json   # 训练曲线数据
    └──── competition_results_go.json # 对战结果
```

### 2.2 成语数据模块（idiom_data.py）

**核心类：IdiomDictionary**

```python
class IdiomDictionary:
    def __init__(self, use_pinyin: bool = False):
        # 核心数据结构
        self.idioms: Dict[int, Tuple[str, Optional[str]]] = {}  # id -> (text, pinyin)
        
        # 索引结构（关键优化）
        self.head_index: Dict[str, List[int]] = defaultdict(list)  # 首字 -> 成语ID列表
        self.tail_index: Dict[str, List[int]] = defaultdict(list)  # 尾字 -> 成语ID列表
        
        # 音同索引（可选）
        self.head_pinyin_index: Dict[str, List[int]] = defaultdict(list)
        self.tail_pinyin_index: Dict[str, List[int]] = defaultdict(list)
    
    def get_followers(self, idiom_id: int, use_pinyin: bool = False) -> List[int]:
        """获取可接龙的成语列表（O(1) 索引查询）"""
        text = self.id_to_text[idiom_id]
        tail_char = text[-1]
        
        # 字面匹配
        followers = set(self.head_index.get(tail_char, []))
        
        # 音同匹配（可选）
        if use_pinyin and self.use_pinyin:
            pinyin = self.id_to_pinyin[idiom_id]
            if pinyin:
                tail_pinyin = pinyin.split()[-1]
                followers.update(self.head_pinyin_index.get(tail_pinyin, []))
        
        return list(followers)
```

**设计要点**：
- 使用 `head_index` 和 `tail_index` 实现首字/尾字的 O(1) 查询
- 支持字面匹配和音同匹配两种模式
- ID 映射避免字符串比较开销

### 2.3 图数据结构（idiom_graph.py）

**核心类：IdiomGraph**

```python
class IdiomGraph:
    def __init__(self, dictionary: IdiomDictionary, use_pinyin: bool = False):
        self.dictionary = dictionary
        
        # 邻接表表示（双向）
        self.adjacency: Dict[int, List[int]] = defaultdict(list)      # 出度邻居
        self.reverse_adjacency: Dict[int, List[int]] = defaultdict(list)  # 入度邻居
        
        self._build_graph()
        self._compute_stats()
    
    def _compute_components(self) -> None:
        """计算弱连通分量（迭代实现避免栈溢出）"""
        visited = set()
        self.components: List[Set[int]] = []
        
        for start_node in self.dictionary.get_all_ids():
            if start_node in visited:
                continue
            
            component = set()
            stack = [start_node]  # 显式栈替代递归
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                
                # 正向边和反向边（弱连通）
                for neighbor in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
                for predecessor in self.reverse_adjacency.get(node, []):
                    if predecessor not in visited:
                        stack.append(predecessor)
            
            self.components.append(component)
```

**核心类：GameState（状态表示）**

```python
class GameState:
    """游戏状态 = (last_idiom, used_set)"""
    
    def __init__(self, last_idiom: Optional[int] = None,
                 used_set: Optional[Set[int]] = None):
        self.last_idiom = last_idiom
        self.used_set = used_set if used_set is not None else set()
    
    def make_move(self, idiom_id: int) -> 'GameState':
        """执行移动，返回新状态（不可变设计）"""
        new_used = self.used_set | {idiom_id}
        return GameState(last_idiom=idiom_id, used_set=new_used)
    
    def get_legal_moves(self, graph: IdiomGraph) -> List[int]:
        """获取合法移动"""
        if self.last_idiom is None:
            # 游戏开始，可选任意成语
            return [id_ for id_ in graph.dictionary.get_all_ids()
                    if id_ not in self.used_set]
        else:
            # 必须接龙
            followers = graph.get_neighbors(self.last_idiom)
            return [id_ for id_ in followers if id_ not in self.used_set]
    
    def to_key(self) -> Tuple[Optional[int], frozenset]:
        """生成缓存键（可哈希）"""
        return (self.last_idiom, frozenset(self.used_set))
```

---

## 三、求解算法详解

### 3.1 Sprague-Grundy 求解器（sg_solver.py）

**理论基础**：
- SG = 0 表示必败态（P-position）
- SG > 0 表示必胜态（N-position）
- 必胜策略：选择移动到 SG = 0 的状态

**核心算法**：

```python
class SGSolver:
    def __init__(self, graph: IdiomGraph, max_cache_size: int = 1000000):
        self.graph = graph
        self.sg_cache: Dict[Tuple[Optional[int], frozenset], int] = {}
    
    def mex(self, s: Set[int]) -> int:
        """Minimum Excludant: 集合中未出现的最小非负整数"""
        i = 0
        while i in s:
            i += 1
        return i
    
    def calculate_sg(self, state: GameState) -> int:
        """计算状态的 SG 值（带记忆化）"""
        state_key = state.to_key()
        
        # 缓存检查
        if state_key in self.sg_cache:
            self.cache_hits += 1
            return self.sg_cache[state_key]
        
        moves = state.get_legal_moves(self.graph)
        
        # 终止状态：无法移动，SG = 0
        if not moves:
            self._add_to_cache(state_key, 0)
            return 0
        
        # 计算所有后继状态的 SG 值
        successor_sg_values: Set[int] = set()
        for move in moves:
            new_state = state.make_move(move)
            sg = self.calculate_sg(new_state)
            successor_sg_values.add(sg)
        
        # 计算 mex
        sg_value = self.mex(successor_sg_values)
        self._add_to_cache(state_key, sg_value)
        
        return sg_value
```

**复杂度瓶颈**：
朴素状态枚举的上界为 $O(|V| \cdot 2^{|V|})$。对于 $|V|=26,108$，直接枚举不在常规计算预算内。当前实现对部分 200–300 节点连通子图未在预设预算内完成；这反映实现与预算限制，并非一般不可解性证明。

### 3.2 Minimax + Alpha-Beta 求解器（minimax_solver.py）

```python
class MinimaxSolver:
    def minimax(self, state: GameState, depth: int = 0,
                alpha: float = -float('inf'), beta: float = float('inf'),
                is_maximizing: bool = True) -> int:
        """Minimax + Alpha-Beta 剪枝"""
        state_key = state.to_key()
        
        if state_key in self.value_cache:
            return self.value_cache[state_key]
        
        moves = state.get_legal_moves(self.graph)
        
        # 终止状态
        if not moves:
            value = -1 if is_maximizing else 1
            self._add_to_cache(state_key, value)
            return value
        
        # 启发式排序（提高剪枝效率）
        moves = self._order_moves(state, moves, is_maximizing)
        
        if is_maximizing:
            best_value = -float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value = self.minimax(new_state, depth + 1, alpha, beta, False)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                
                if alpha >= beta:
                    self.beta_cutoffs += 1
                    break  # Beta 剪枝
            return best_value
        else:
            # MIN 层...
```

**启发式排序策略**：
- MAX 层：优先选择出度小的成语（让对手选择少）
- MIN 层：优先选择出度大的成语（给对手更多选择）

### 3.3 Value Iteration 求解器（q_solver.py）

**核心思想**：不考虑 `used_set`，学习每个成语的"固有价值"

```python
class ValueIterationSolver:
    def solve(self):
        """运行价值迭代算法"""
        for i in range(self.iterations):
            new_values = {}
            delta = 0.0
            
            for u in all_nodes:
                neighbors = neighbors_map[u]
                
                if not neighbors:
                    new_values[u] = -1.0  # 无路可走，必败
                    continue
                
                # 价值公式: V(u) = max(1 (如果 v 是死路) 或 -gamma * V(v))
                best_val = -float('inf')
                for v in neighbors:
                    if not neighbors_map[v]:
                        val = 1.0  # 对手无路可走，我赢
                    else:
                        val = -self.gamma * self.values[v]
                    
                    best_val = max(best_val, val)
                
                new_values[u] = best_val
                delta = max(delta, abs(new_values[u] - self.values[u]))
            
            self.values = new_values
            
            if delta < 1e-4:  # 收敛检查
                break
```

**局限性**：忽略 `used_set` 导致无法处理"成语不可重复"规则。

### 3.4 Q-Learning 自对抗求解器（selfplay_solver.py）

**状态空间压缩**：
将完整的 MDP 简化为 POMDP：
- **状态**：$s = u \in V$（当前成语）
- **动作**：$a = v \in A(u)$（下一个成语）
- **Q 值**：$Q(u, v)$ 表示从 $u$ 走到 $v$ 的期望长期胜率

```python
class SelfPlaySolver:
    def __init__(self, graph: IdiomGraph, lr=0.05, gamma=0.95, epsilon=0.3):
        self.q_table = defaultdict(dict)
        
        # 初始化Q值
        for u in graph.dictionary.get_all_ids():
            for v in graph.get_neighbors(u):
                self.q_table[u][v] = random.uniform(-0.1, 0.1)
    
    def play_episode(self, max_steps=500):
        """模拟一局自对抗游戏"""
        current_id = random.choice(all_nodes)
        used_set = {current_id}
        history = []  # [(player, from, to), ...]
        current_player = 0
        
        for step in range(max_steps):
            next_id = self.choose_action(current_id, used_set, training=True)
            
            if next_id is None:
                winner = 1 - current_player
                return history, winner
            
            history.append((current_player, current_id, next_id))
            used_set.add(next_id)
            current_id = next_id
            current_player = 1 - current_player
    
    def update_q(self, history, winner):
        """逆向传播更新Q值"""
        n = len(history)
        for i in range(n - 1, -1, -1):
            player, from_id, to_id = history[i]
            
            # 计算奖励
            if i == n - 1:
                reward = 1.0 if winner == player else -1.0
            else:
                reward = 0.5 if winner == player else -0.5
            
            old_q = self.q_table[from_id].get(to_id, 0.0)
            
            # 计算下一状态的最大Q值
            if i < n - 1:
                next_from = history[i + 1][1]
                next_max_q = max(self.q_table[next_from].values())
            else:
                next_max_q = 0.0
            
            # TD 更新：零和博弈版本
            new_q = old_q + self.lr * (reward + self.gamma * (-next_max_q) - old_q)
            self.q_table[from_id][to_id] = new_q
```

**关键公式**：
$$Q(u, v) \leftarrow Q(u, v) + \alpha \left[ R + \gamma \cdot (-\max_{v'} Q(v, v')) - Q(u, v) \right]$$

注意 `Target = -\max_{v'} Q(v, v')`，取负号是因为零和博弈中对手的最大 Q 值即为我方的最小期望收益。

---

## 四、Go 语言高性能实现

### 4.1 图构建与剪枝（idiom.go）

```go
type Graph struct {
    N    int
    Text []string
    Adj  [][]int
}

func LoadGraph(filepath string, usePinyin bool) *Graph {
    // 解析 JSON
    var items []map[string]interface{}
    json.Unmarshal(data, &items)
    
    // 构建首字索引
    headIndex := make(map[string][]int)
    for i, idm := range idioms {
        runes := []rune(idm.text)
        head := string(runes[0])
        headIndex[head] = append(headIndex[head], i)
    }
    
    // 构建邻接表
    for i, idm := range idioms {
        tailChar := string([]rune(idm.text)[3])
        followers := make(map[int]bool)
        for _, fid := range headIndex[tailChar] {
            followers[fid] = true
        }
        delete(followers, i)  // 删除自环
        g.Adj[i] = make([]int, 0, len(followers))
        for fid := range followers {
            g.Adj[i] = append(g.Adj[i], fid)
        }
    }
    return g
}
```

### 4.2 密集采样训练（train_dense.go）

**训练框架**：8 核并行 + 每 10 万局评估

```go
func main() {
    // 训练参数
    totalEpisodes := 5_000_000   // 500万局
    evalEvery := 100_000         // 每10万局评估
    lr := 0.05                   // 学习率
    gamma := 0.85                // 折扣因子
    epsStart := 0.3              // 初始探索率
    numCPU := runtime.NumCPU()   // 8核
    
    // 初始化Q表
    qTable := make([]map[int]float64, g.N)
    for i := range g.N {
        qTable[i] = make(map[int]float64)
        for _, v := range g.Adj[i] {
            qTable[i][v] = rand.Float64()*0.2 - 0.1
        }
    }
    
    // 每轮训练：多核并行采样
    for ep := evalEvery; ep <= totalEpisodes; ep += evalEvery {
        pwDelta := deltaEp / numCPU  // 每核训练局数
        
        var wg sync.WaitGroup
        for w := 0; w < numCPU; w++ {
            wg.Add(1)
            go func(wid int) {
                defer wg.Done()
                // 本地训练...
                for e := 0; e < pwDelta; e++ {
                    // 模拟一局
                    // 更新本地Q表副本
                }
            }(w)
        }
        wg.Wait()
        
        // Q 表聚合（算术平均）
        for i := range qTable {
            for v := range qTable[i] {
                var sum float64
                for w := 0; w < numCPU; w++ {
                    sum += results[w].qTable[i][v]
                }
                qTable[i][v] = sum / float64(numCPU)
            }
        }
        
        // 评估胜率
        winRate := evalFunc(qTable, ep)
        fmt.Printf("  %-8d | %-8.2f | %s\n", ep, winRate, stage)
    }
}
```

**训练动态参数**：

```go
// 探索率衰减
eps := epsStart * (1 - progress) + 0.05

// 学习率衰减（后期精确收敛）
currentLR := lr * (1.0 - progress*0.9 + 0.1)

// 混合对手：30% 对抗随机
opponentIsRandom := rng.Float64() < 0.3
```

**TD 更新（逆向传播）**：

```go
// 游戏结束时逆向更新
for i := len(history) - 1; i >= 0; i-- {
    m := history[i]
    
    // 计算奖励
    var reward float64
    if i == len(history)-1 {
        if m.player == winner { reward = 1.0 } else { reward = -1.0 }
    } else {
        if m.player == winner { reward = 0.5 } else { reward = -0.5 }
    }
    
    // TD 更新
    oldQ := local.qTable[m.from][m.to]
    var nextMax float64
    if i < len(history)-1 {
        nf := history[i+1].to
        for _, nv := range g.Adj[nf] {
            if q := local.qTable[nf][nv]; q > nextMax { nextMax = q }
        }
    }
    
    newQ := oldQ + currentLR*(reward + gamma*(-nextMax) - oldQ)
    local.qTable[m.from][m.to] = newQ
}
```

---

## 五、实验结果

### 5.1 训练曲线

基于 500 万局训练的 51 个评估点：

| 训练阶段 | 局数范围 | 胜率 | 特征 |
|:---|:---:|:---:|:---|
| 随机初始化 | 0 局 | 47.8% | Q 表为小随机值 |
| 快速学习期 | 0 – 10 万局 | 47.8% → 89.3% | 识别基本必胜/必败模式 |
| 稳定提升期 | 10 万 – 100 万局 | 89.3% → 92.5% | 学习长程策略 |
| 收敛期 | 100 万 – 500 万局 | 均值 95.3% (σ=1.09%) | Q 表趋于稳定 |

### 5.2 最终对战评估

| 对战组合 | Q-Learning 胜率 | 对手胜率 | 总局数 |
|:---|:---:|:---:|:---:|
| **QL vs Random** | **97.4%** | 2.6% | 10,000 |

### 5.3 超参数配置

| 超参数 | 符号 | 数值 | 说明 |
|:---|:---:|:---:|:---|
| 总训练局数 | $N$ | 5,000,000 | 大规模训练 |
| 学习率 | $\alpha$ | 0.05 → 0.005 | 线性衰减 |
| 折扣因子 | $\gamma$ | 0.85 | 降低远期噪声 |
| 初始探索率 | $\epsilon_{\text{start}}$ | 0.30 | 充分探索 |
| 最小探索率 | $\epsilon_{\text{min}}$ | 0.05 | 保持探索 |
| 混合对手比例 | $\rho_{\text{rand}}$ | 0.30 | 30% 随机对抗 |
| 并行 Worker 数 | $W$ | 8 | 多核加速 |

---

## 六、代码设计亮点与局限性

### 6.1 设计亮点

1. **双向索引优化**：`head_index` 和 `tail_index` 实现 O(1) 接龙查询
2. **不可变状态设计**：`GameState.make_move()` 返回新对象，避免副作用
3. **缓存淘汰策略**：SG/Minimax 求解器的 LRU 缓存控制内存
4. **多核并行架构**：Go 语言 8 核 Worker + Q 表聚合
5. **混合训练策略**：70% 自对抗 + 30% 随机对手，提升泛化性
6. **迭代式连通分量计算**：显式栈替代递归，避免栈溢出

### 6.2 当前局限性

1. **POMDP 近似的误差**：
   - Q 表假设 $Q(u,v)$ 是静态的
   - 实际博弈中，边的好坏取决于 $U_t$（已使用集合）
   - 无法区分"B 已用"和"B 未用"的状态

2. **长尾效应**：
   - 26,108 个节点中，部分边缘节点访问频次不足 10 次
   - Q 表在罕见状态的估值存在偏差

3. **胜率未达 100%**：
   - 当前最佳胜率：97.4%
   - 失败案例归因：马尔可夫假设局限 + 随机对手的"乱拳"效应

---

## 七、单元测试覆盖

```python
class TestAlgorithmConsistency(unittest.TestCase):
    def test_sg_minimax_consistency_small(self):
        """小规模一致性测试"""
        for n in [10, 20, 30]:
            # SG 和 Minimax 必胜判定应一致
            sg_win = sg_solver.calculate_sg_initial() > 0
            mm_win = mm_solver.is_winning(GameState())[0]
            self.assertEqual(sg_win, mm_win)
    
    def test_winning_strategy_validity(self):
        """必胜策略有效性测试"""
        if initial_sg > 0:
            best_move = solver.find_best_move(GameState())
            new_state = GameState().make_move(best_move[0])
            new_sg = solver.calculate_sg(new_state)
            # 必胜策略应移动到 SG=0 的状态
            self.assertEqual(new_sg, 0)
```

---

## 八、待改进方向（供研讨）

1. **显式状态输入**：将 $U_t$ 的压缩表示（Bloom Filter / 位图）作为神经网络输入
2. **蒙特卡洛树搜索 (MCTS)**：在 Q-Learning 先验引导下进行实时搜索（类似 AlphaGo）
3. **图神经网络 (GNN)**：学习成语节点的结构特征
4. **分层 Q 表**：按尾字分组，每组独立 Q 表
5. **动态难度训练**：从简单子图逐步扩展到完整图

---

**文档生成时间**：2026年5月3日
**项目版本**：基于当前代码库状态
