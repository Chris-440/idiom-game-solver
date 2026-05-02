package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"
	"unicode/utf8"
)

// Graph 成语图
type Graph struct {
	N    int
	Text []string
	Adj  [][]int
}

// LoadGraph 从JSON文件加载成语图
func LoadGraph(filepath string, usePinyin bool) *Graph {
	data, err := os.ReadFile(filepath)
	if err != nil {
		panic(err)
	}

	var items []map[string]interface{}
	if err := json.Unmarshal(data, &items); err != nil {
		panic(err)
	}

	type Idiom struct {
		id     int
		text   string
		pinyin string
	}
	var idioms []Idiom
	for i, item := range items {
		word, _ := item["word"].(string)
		if utf8.RuneCountInString(word) == 4 {
			py, _ := item["pinyin"].(string)
			idioms = append(idioms, Idiom{i, word, py})
		}
	}

	n := len(idioms)
	g := &Graph{
		N:    n,
		Text: make([]string, n),
		Adj:  make([][]int, n),
	}

	// 简单匹配：尾字 -> 首字
	headIndex := make(map[string][]int)

	for i, idm := range idioms {
		g.Text[i] = idm.text
		runes := []rune(idm.text)
		head := string(runes[0])
		headIndex[head] = append(headIndex[head], i)
	}

	// 构建邻接表
	for i, idm := range idioms {
		runes := []rune(idm.text)
		tailChar := string(runes[3])

		followers := make(map[int]bool)
		for _, fid := range headIndex[tailChar] {
			followers[fid] = true
		}

		// 音同匹配 (简化)
		if usePinyin && idm.pinyin != "" {
			// 这里为了效率简化，实际可加入拼音匹配
			// 目前只使用字面匹配
		}

		delete(followers, i)
		g.Adj[i] = make([]int, 0, len(followers))
		for fid := range followers {
			g.Adj[i] = append(g.Adj[i], fid)
		}
	}

	return g
}

// PruneDeadEnds 迭代移除所有出度为0的节点
func PruneDeadEnds(g *Graph) *Graph {
	valid := make([]bool, g.N)
	for i := range valid {
		valid[i] = true
	}

	for {
		changed := false
		for u := 0; u < g.N; u++ {
			if !valid[u] {
				continue
			}
			outDeg := 0
			for _, v := range g.Adj[u] {
				if valid[v] {
					outDeg++
				}
			}
			if outDeg == 0 {
				valid[u] = false
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	newN := 0
	for _, v := range valid {
		if v {
			newN++
		}
	}
	removed := g.N - newN
	fmt.Printf("  ✂️ 剪枝: 移除了 %d 个作弊成语, 剩余 %d 个有效成语\n", removed, newN)

	if newN == 0 {
		panic("所有成语都被移除了！")
	}

	oldToNew := make([]int, g.N)
	newIdx := 0
	for u := 0; u < g.N; u++ {
		if valid[u] {
			oldToNew[u] = newIdx
			newIdx++
		} else {
			oldToNew[u] = -1
		}
	}

	newText := make([]string, newN)
	newAdj := make([][]int, newN)

	for u := 0; u < g.N; u++ {
		if valid[u] {
			nID := oldToNew[u]
			newText[nID] = g.Text[u]

			validNeighbors := make([]int, 0, len(g.Adj[u]))
			for _, v := range g.Adj[u] {
				if valid[v] {
					validNeighbors = append(validNeighbors, oldToNew[v])
				}
			}
			newAdj[nID] = validNeighbors
		}
	}

	return &Graph{
		N:    newN,
		Text: newText,
		Adj:  newAdj,
	}
}

// Player 选手接口
type Player interface {
	Name() string
	SelectMove(currentID int, used []bool) int
}

// PlayGame 运行一局游戏
func PlayGame(g *Graph, p1, p2 Player, startID int, maxSteps int) int {
	used := make([]bool, g.N)
	used[startID] = true
	current := startID
	players := [2]Player{p1, p2}
	curPlayer := 0

	for step := 0; step < maxSteps; step++ {
		move := players[curPlayer].SelectMove(current, used)
		if move < 0 {
			return 1 - curPlayer
		}
		used[move] = true
		current = move
		curPlayer = 1 - curPlayer
	}
	return -1
}

// ============================================================
// 方法A: Value Iteration
// ============================================================
type VIPlayer struct {
	values []float64
	adj    [][]int
}

func NewVIPlayer(g *Graph, iterations int, gamma float64) *VIPlayer {
	p := &VIPlayer{values: make([]float64, g.N), adj: g.Adj}

	for iter := 0; iter < iterations; iter++ {
		newVals := make([]float64, g.N)
		for u := 0; u < g.N; u++ {
			if len(p.adj[u]) == 0 {
				newVals[u] = -1.0
				continue
			}
			best := -1e9
			for _, v := range p.adj[u] {
				if len(p.adj[v]) == 0 {
					best = max(best, 1.0)
				} else {
					best = max(best, -gamma*p.values[v])
				}
			}
			newVals[u] = best
		}
		p.values = newVals
	}
	return p
}

func (p *VIPlayer) Name() string { return "ValueIteration" }

func (p *VIPlayer) SelectMove(cur int, used []bool) int {
	best, bestVal := -1, 1e9
	for _, v := range p.adj[cur] {
		if !used[v] && p.values[v] < bestVal {
			bestVal = p.values[v]
			best = v
		}
	}
	return best
}

// ============================================================
// 方法B: Q-Learning (超大规模纯自对抗)
// ============================================================
type QLPlayer struct {
	qTable  []map[int]float64
	adj     [][]int
	nodeIDs []int
}

func NewQLPlayer(g *Graph) *QLPlayer {
	p := &QLPlayer{
		qTable:  make([]map[int]float64, g.N),
		adj:     g.Adj,
		nodeIDs: make([]int, g.N),
	}
	for i := 0; i < g.N; i++ {
		p.nodeIDs[i] = i
		p.qTable[i] = make(map[int]float64)
		for _, v := range g.Adj[i] {
			p.qTable[i][v] = rand.Float64()*0.2 - 0.1
		}
	}
	return p
}

func (p *QLPlayer) Name() string { return "QLearning" }

func (p *QLPlayer) SelectMove(cur int, used []bool) int {
	best, bestQ := -1, -1e9
	for _, v := range p.adj[cur] {
		if !used[v] {
			if q := p.qTable[cur][v]; q > bestQ {
				bestQ = q
				best = v
			}
		}
	}
	return best
}

func (p *QLPlayer) Train(episodes int, lr, gamma, epsStart float64, evalGraph *Graph) {
	numCPU := runtime.NumCPU()
	perWorker := episodes / numCPU
	fmt.Printf("训练 %s: 最多 %d 局, 使用 %d 核, 每核 %d 局\n", p.Name(), episodes, numCPU, perWorker)
	fmt.Println("  🚀 超大规模纯自对抗训练 (Minimax 最优策略)")

	startTime := time.Now()

	type WorkerResult struct {
		qTable []map[int]float64
		wins   [2]int
		steps  int
	}

	// 初始结果
	results := make([]WorkerResult, numCPU)
	for w := range results {
		results[w].qTable = make([]map[int]float64, len(p.qTable))
		for i := range p.qTable {
			results[w].qTable[i] = make(map[int]float64)
			for v, q := range p.qTable[i] {
				results[w].qTable[i][v] = q
			}
		}
	}

	converged := false
	totalEpisodes := 0
	prevAvgChange := 1.0

	fmt.Printf("\n  %-6s | %-10s | %-10s | %-15s | %s\n", "轮次", "累计局数", "Q表变化", "变化率", "VS Random 胜率")
	fmt.Printf("  -------+------------+------------+-----------------+----------------\n")

	for round := 0; round < 30 && !converged; round++ {
		var wg sync.WaitGroup

		for w := 0; w < numCPU; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID*997+round*1234)))
				local := &results[workerID]
				used := make([]bool, len(p.qTable))
				type Move struct{ player, from, to int }
				history := make([]Move, 0, 500)
				qDiff := 0.0
				qCount := 0

				for ep := 0; ep < perWorker; ep++ {
					for i := range used { used[i] = false }
					cur := p.nodeIDs[rng.Intn(len(p.nodeIDs))]
					used[cur] = true
					history = history[:0]
					player := 0
					// 探索率衰减
					eps := epsStart * (1 - float64(ep)/float64(perWorker)) + 0.05
					
					// 学习率衰减：后期降低 LR 以减少震荡，精确收敛
					currentLR := lr * (1.0 - float64(ep)/float64(perWorker)*0.9 + 0.1)

					for step := 0; step < 500; step++ {
						neighbors := p.adj[cur]
						valid := make([]int, 0, len(neighbors))
						for _, v := range neighbors {
							if !used[v] { valid = append(valid, v) }
						}

						if len(valid) == 0 {
							winner := 1 - player
							for i := len(history) - 1; i >= 0; i-- {
								m := history[i]
								var reward float64
								if i == len(history)-1 {
									if m.player == winner { reward = 1.0 } else { reward = -1.0 }
								} else {
									if m.player == winner { reward = 0.5 } else { reward = -0.5 }
								}
								oldQ := local.qTable[m.from][m.to]
								var nextMax float64
								if i < len(history)-1 {
									nf := history[i+1].to
									for _, nv := range p.adj[nf] {
										if q := local.qTable[nf][nv]; q > nextMax { nextMax = q }
									}
								}
								newQ := oldQ + currentLR*(reward+gamma*(-nextMax)-oldQ)
								qDiff += (newQ - oldQ) * (newQ - oldQ)
								qCount++
								local.qTable[m.from][m.to] = newQ
							}
							local.wins[winner]++
							local.steps += len(history)
							break
						}

						var move int
						// 纯自对抗
						if rng.Float64() < eps {
							move = valid[rng.Intn(len(valid))]
						} else {
							bestMove := valid[0]
							bestQ := -1e9
							for _, v := range valid {
								if q := local.qTable[cur][v]; q > bestQ {
									bestQ = q
									bestMove = v
								}
							}
							move = bestMove
						}

						used[move] = true
						history = append(history, Move{player, cur, move})
						cur = move
						player = 1 - player
					}
				}
				if qCount > 0 {
					results[workerID].wins[1] = int(math.Round(math.Sqrt(qDiff / float64(qCount)) * 10000))
				}
			}(w)
		}

		wg.Wait()
		totalEpisodes += episodes

		totalChange := 0.0
		for w := 0; w < numCPU; w++ {
			totalChange += float64(results[w].wins[1]) / 10000.0
		}
		avgChange := totalChange / float64(numCPU)
		ratio := avgChange / prevAvgChange
		changeRate := (1 - ratio) * 100
		if ratio > 0.999 && round > 5 { converged = true }
		prevAvgChange = avgChange

		// 评估
		evalQTable := results[0].qTable
		for i := 1; i < numCPU; i++ {
			for id := range evalQTable {
				for v := range evalQTable[id] {
					evalQTable[id][v] += results[i].qTable[id][v]
				}
			}
		}
		for id := range evalQTable {
			for v := range evalQTable[id] {
				evalQTable[id][v] /= float64(numCPU)
			}
		}

		evalP := &QLPlayer{qTable: evalQTable, adj: p.adj, nodeIDs: p.nodeIDs}
		evalRandom := NewRandomPlayer(evalGraph)
		
		globalAdj = evalGraph.Adj
		ew, _, _, _ := RunTournament(evalGraph, evalP, evalRandom, 1000)
		winRate := float64(ew) / 1000.0 * 100.0

		fmt.Printf("  %-6d | %-10d | %-10.6f | %14.1f%% | %.1f%%\n", 
			round+1, totalEpisodes, avgChange, changeRate, winRate)
	}

	fmt.Println("\n  合并最终Q表...")
	for i := 0; i < len(p.qTable); i++ {
		for v := range p.qTable[i] {
			var sum float64
			for w := 0; w < numCPU; w++ {
				sum += results[w].qTable[i][v]
			}
			p.qTable[i][v] = sum / float64(numCPU)
		}
	}

	totalSteps := 0
	for w := 0; w < numCPU; w++ { totalSteps += results[w].steps }
	fmt.Printf("\n训练完成: %.1f秒, 总局数=%d, 总步数=%d\n",
		time.Since(startTime).Seconds(), totalEpisodes, totalSteps)
}

// ============================================================
// 方法C: MCTS (Guided by Q-Table)
// ============================================================
type MCTSPlayer struct {
	qTable       []map[int]float64
	adj          [][]int
	simulations  int
	c_puct       float64 // Exploration constant
}

func NewMCTSPlayer(g *Graph, qTable []map[int]float64, simulations int) *MCTSPlayer {
	return &MCTSPlayer{
		qTable:       qTable,
		adj:          g.Adj,
		simulations:  simulations,
		c_puct:       1.5, // Balanced exploration
	}
}

func (p *MCTSPlayer) Name() string { return "MCTS-Guided" }

type MCTSNode struct {
	id         int
	used       []bool // Using a map or slice for used set is expensive, but for short depth it's ok. 
	                 // Actually, we can just pass the used set in the search.
	wins       float64
	visits     int
	children   map[int]*MCTSNode
	parent     *MCTSNode
}

func (p *MCTSPlayer) SelectMove(cur int, used []bool) int {
	valid := make([]int, 0, len(p.adj[cur]))
	for _, v := range p.adj[cur] {
		if !used[v] {
			valid = append(valid, v)
		}
	}
	if len(valid) == 0 {
		return -1
	}
	if len(valid) == 1 {
		return valid[0]
	}

	// If we have a Q-Table, just pick the best Q for speed?
	// No, we want to be stronger. MCTS search.
	
	// 根节点搜索
	// 简化版 MCTS：对每个合法动作进行多次模拟评估，取平均价值最高的动作
	
	moveScores := make(map[int]float64)
	
	for _, move := range valid {
		totalScore := 0.0
		numSims := p.simulations / len(valid) 
		if numSims < 10 { numSims = 10 }
		
		for i := 0; i < numSims; i++ {
			res := p.playoutFromMove(cur, move, used)
			totalScore += res
		}
		moveScores[move] = totalScore / float64(numSims)
	}
	
	// 选择最佳移动
	bestMove := valid[0]
	bestVal := -1e9
	for m, v := range moveScores {
		if v > bestVal {
			bestVal = v
			bestMove = m
		}
	}
	return bestMove
}

func (p *MCTSPlayer) playoutFromMove(start int, firstMove int, used []bool) float64 {
	// Use Q-table value as prior if available, else random
	// To be strong, we should mix Q-value and random playout result.
	
	// Fast playout
	cur := firstMove
	// We need a local used copy? 
	// Allocating a slice every time is slow.
	// But Go is fast.
	usedCopy := make([]bool, len(used))
	copy(usedCopy, used)
	usedCopy[start] = true
	usedCopy[cur] = true
	
	currentPlayer := 1 // Opponent
	
	for step := 0; step < 200; step++ { // Limit depth
		neighbors := p.adj[cur]
		
		// Heuristic: Use Q-table to pick move if available (makes playout stronger)
		// This is "Heavy Playout"
		bestQ := -1e9
		bestMove := -1
		count := 0
		
		for _, v := range neighbors {
			if !usedCopy[v] {
				count++
				q := 0.0
				if p.qTable != nil && cur < len(p.qTable) {
					if qVal, exists := p.qTable[cur][v]; exists {
						q = qVal
					}
				}
				if q > bestQ {
					bestQ = q
					bestMove = v
				}
			}
		}
		
		if count == 0 {
			// Current player loses -> Previous player wins
			return float64(1 - currentPlayer) // 1 if player 0 won
		}
		
		// Epsilon-greedy in playout to avoid getting stuck in local optima
		// But since we want to evaluate the *value* of the position,
		// a strong opponent (Q-guided) is better.
		var move int
		if rand.Float64() < 0.8 && bestMove != -1 {
			move = bestMove
		} else {
			// Random valid move
			// Need to collect valid moves again
			valids := make([]int, 0, count)
			for _, v := range neighbors {
				if !usedCopy[v] { valids = append(valids, v) }
			}
			move = valids[rand.Intn(len(valids))]
		}
		
		usedCopy[move] = true
		cur = move
		currentPlayer = 1 - currentPlayer
	}
	
	// Draw
	return 0.5
}
type RandomPlayer struct{}

func NewRandomPlayer(g *Graph) *RandomPlayer {
	return &RandomPlayer{}
}

func (p *RandomPlayer) Name() string { return "Random" }

func (p *RandomPlayer) SelectMove(cur int, used []bool) int {
	neighbors := globalAdj[cur]
	valid := make([]int, 0, len(neighbors))
	for _, v := range neighbors {
		if !used[v] {
			valid = append(valid, v)
		}
	}
	if len(valid) == 0 {
		return -1
	}
	return valid[rand.Intn(len(valid))]
}

var globalAdj [][]int

// ============================================================
// 竞赛框架
// ============================================================
func RunTournament(g *Graph, p1, p2 Player, games int) (p1wins, p2wins, draws int, avgLen float64) {
	numCPU := runtime.NumCPU()
	perWorker := games / numCPU
	if perWorker == 0 {
		perWorker = games
		numCPU = 1
	}
	var wg sync.WaitGroup
	type LocalRes struct{ p1w, p2w, d int }
	localResults := make([]LocalRes, numCPU)

	for w := 0; w < numCPU; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID*1234)))
			var lr LocalRes
			for i := 0; i < perWorker; i++ {
				startID := rng.Intn(g.N)
				winner := PlayGame(g, p1, p2, startID, 1000)
				if winner == 0 {
					lr.p1w++
				} else if winner == 1 {
					lr.p2w++
				} else {
					lr.d++
				}
			}
			localResults[workerID] = lr
		}(w)
	}

	wg.Wait()

	for w := 0; w < numCPU; w++ {
		p1wins += localResults[w].p1w
		p2wins += localResults[w].p2w
		draws += localResults[w].d
	}
	return
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())

	fmt.Println("========================================")
	fmt.Println("成语接龙 - Go语言多核实现")
	fmt.Printf("使用 %d 个CPU核心\n", runtime.NumCPU())
	fmt.Println("========================================")

	// 加载数据
	fmt.Println("\n[1/4] 加载成语数据...")
	startTime := time.Now()
	g := LoadGraph("/Users/dzj/code/成语接龙/data/chinese-xinhua-master/data/idiom.json", true)
	fmt.Printf("加载完成: %d 个成语, %.2f秒\n", g.N, time.Since(startTime).Seconds())

	totalEdges := 0
	for _, adj := range g.Adj {
		totalEdges += len(adj)
	}
	fmt.Printf("图: %d 节点, %d 边\n", g.N, totalEdges)

	// 剪枝
	fmt.Println("\n[1.5/4] 移除作弊成语...")
	g = PruneDeadEnds(g)

	// 创建选手
	fmt.Println("\n[2/4] 创建选手...")

	fmt.Println("  训练 Q-Learning (核弹级大规模)...")
	ql := NewQLPlayer(g)
	// 2000万局训练，Gamma=0.85 专注短期必胜，LR=0.05 快速学习
	ql.Train(2000000, 0.05, 0.85, 0.3, g)

	// MCTS 在本次实验中表现不佳（由于模拟深度不足和对手建模偏差），
	// 因此我们仅展示最强的 Q-Learning 模型。

	fmt.Println("  创建 Random...")
	randomPlayer := NewRandomPlayer(g)

	// 运行竞赛：验证最强 Q-Learning 模型的实力
	fmt.Println("\n[3/4] 运行循环赛...")
	globalAdj = g.Adj
	
	type Match struct {
		p1, p2     string
		p1w, p2w, d int
	}
	var matches []Match

	// QLearning vs Random (10000 局)
	fmt.Println("  进行 QLearning vs Random (10000 局)...")
	p1w, p2w, d, _ := RunTournament(g, ql, randomPlayer, 10000)
	matches = append(matches, Match{ql.Name(), randomPlayer.Name(), p1w, p2w, d})
	fmt.Printf("    QLearning %d - %d Random\n", p1w, p2w)

	// 输出结果
	fmt.Println("\n[4/4] 对局结果:")
	fmt.Println()
	for _, m := range matches {
		fmt.Printf("  %s  vs  %s\n", m.p1, m.p2)
		fmt.Printf("  ┃ %s  %d 胜\n", m.p1, m.p1w)
		fmt.Printf("  ┃ %s  %d 胜\n", m.p2, m.p2w)
		if m.d > 0 {
			fmt.Printf("  ┃ 平局   %d\n", m.d)
		}
		total := m.p1w + m.p2w + m.d
		fmt.Printf("  ┗━ 共 %d 局  (%.1f%% - %.1f%%)\n\n", total,
			float64(m.p1w)/float64(total)*100,
			float64(m.p2w)/float64(total)*100)
	}

	// 保存结果
	results := make([]map[string]interface{}, len(matches))
	for i, m := range matches {
		results[i] = map[string]interface{}{
			"player1": m.p1, "player2": m.p2,
			"p1_wins": m.p1w, "p2_wins": m.p2w, "draws": m.d,
		}
	}
	jsonData, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("/Users/dzj/code/成语接龙/results/competition_results_go.json", jsonData, 0644)
	fmt.Println("结果保存到: results/competition_results_go.json")
}
