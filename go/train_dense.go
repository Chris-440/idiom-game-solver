package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
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

func LoadGraph(filepath string) *Graph {
	data, err := os.ReadFile(filepath)
	if err != nil { panic(err) }
	var items []map[string]interface{}
	if err := json.Unmarshal(data, &items); err != nil { panic(err) }

	type Idiom struct { id int; text string }
	var idioms []Idiom
	for i, item := range items {
		word, _ := item["word"].(string)
		if utf8.RuneCountInString(word) == 4 {
			idioms = append(idioms, Idiom{i, word})
		}
	}

	n := len(idioms)
	g := &Graph{N: n, Text: make([]string, n), Adj: make([][]int, n)}
	headIndex := make(map[string][]int)
	for i, idm := range idioms {
		g.Text[i] = idm.text
		runes := []rune(idm.text)
		head := string(runes[0])
		headIndex[head] = append(headIndex[head], i)
	}
	for i, idm := range idioms {
		runes := []rune(idm.text)
		tailChar := string(runes[3])
		followers := make(map[int]bool)
		for _, fid := range headIndex[tailChar] { followers[fid] = true }
		delete(followers, i)
		g.Adj[i] = make([]int, 0, len(followers))
		for fid := range followers { g.Adj[i] = append(g.Adj[i], fid) }
	}
	return g
}

func PruneDeadEnds(g *Graph) *Graph {
	valid := make([]bool, g.N)
	for i := range valid { valid[i] = true }
	for {
		changed := false
		for u := 0; u < g.N; u++ {
			if !valid[u] { continue }
			outDeg := 0
			for _, v := range g.Adj[u] { if valid[v] { outDeg++ } }
			if outDeg == 0 { valid[u] = false; changed = true }
		}
		if !changed { break }
	}
	newN := 0
	for _, v := range valid { if v { newN++ } }
	removed := g.N - newN
	fmt.Printf("  ✂️ 剪枝: 移除 %d 个作弊成语, 剩余 %d 个\n", removed, newN)
	oldToNew := make([]int, g.N)
	newIdx := 0
	for u := 0; u < g.N; u++ {
		if valid[u] { oldToNew[u] = newIdx; newIdx++ } else { oldToNew[u] = -1 }
	}
	newText := make([]string, newN)
	newAdj := make([][]int, newN)
	for u := 0; u < g.N; u++ {
		if valid[u] {
			nID := oldToNew[u]
			newText[nID] = g.Text[u]
			vn := make([]int, 0, len(g.Adj[u]))
			for _, v := range g.Adj[u] { if valid[v] { vn = append(vn, oldToNew[v]) } }
			newAdj[nID] = vn
		}
	}
	return &Graph{N: newN, Text: newText, Adj: newAdj}
}

type Player interface { Name() string; SelectMove(cur int, used []bool) int }

func PlayGame(g *Graph, p1, p2 Player, startID int, maxSteps int) int {
	used := make([]bool, g.N); used[startID] = true
	cur := startID; players := [2]Player{p1, p2}; cp := 0
	for step := 0; step < maxSteps; step++ {
		move := players[cp].SelectMove(cur, used)
		if move < 0 { return 1 - cp }
		used[move] = true; cur = move; cp = 1 - cp
	}
	return -1
}

// ============================================================
// Q-Learning Player
// ============================================================
type QLPlayer struct {
	qTable  []map[int]float64
	adj     [][]int
	nodeIDs []int
}

func NewQLPlayer(g *Graph) *QLPlayer {
	p := &QLPlayer{qTable: make([]map[int]float64, g.N), adj: g.Adj, nodeIDs: make([]int, g.N)}
	for i := 0; i < g.N; i++ {
		p.nodeIDs[i] = i
		p.qTable[i] = make(map[int]float64)
		for _, v := range g.Adj[i] { p.qTable[i][v] = rand.Float64()*0.2 - 0.1 }
	}
	return p
}

func (p *QLPlayer) Name() string { return "QLearning" }

func (p *QLPlayer) SelectMove(cur int, used []bool) int {
	best, bestQ := -1, -1e9
	for _, v := range p.adj[cur] {
		if !used[v] { if q := p.qTable[cur][v]; q > bestQ { bestQ = q; best = v } }
	}
	return best
}

type RandomPlayer struct {
	adj [][]int
}

func (p *RandomPlayer) Name() string { return "Random" }

func NewRandomPlayer(g *Graph) *RandomPlayer { return &RandomPlayer{adj: g.Adj} }

func (p *RandomPlayer) SelectMove(cur int, used []bool) int {
	valid := make([]int, 0, len(p.adj[cur]))
	for _, v := range p.adj[cur] { if !used[v] { valid = append(valid, v) } }
	if len(valid) == 0 { return -1 }
	return valid[rand.Intn(len(valid))]
}

func RunTournament(g *Graph, p1, p2 Player, games int) int {
	numCPU := runtime.NumCPU()
	pw := games / numCPU
	if pw == 0 { pw = games; numCPU = 1 }
	var wg sync.WaitGroup
	localWins := make([]int, numCPU)
	for w := 0; w < numCPU; w++ {
		wg.Add(1)
		go func(wid int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(wid*1234)))
			wp := 0
			for i := 0; i < pw; i++ {
				// 交替先手：偶数局 p1 先手，奇数局 p2 先手
				if i%2 == 0 {
					if PlayGame(g, p1, p2, rng.Intn(g.N), 1000) == 0 { wp++ }
				} else {
					if PlayGame(g, p2, p1, rng.Intn(g.N), 1000) == 1 { wp++ }
				}
			}
			localWins[wid] = wp
		}(w)
	}
	wg.Wait()
	total := 0
	for _, w := range localWins { total += w }
	return total
}

// ============================================================
// 密集采样训练
// ============================================================
func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())

	// 初始化项目路径
	initPaths()

	fmt.Println("========================================")
	fmt.Println("Q-Learning 密集采样训练 (每 10 万局评估一次)")
	fmt.Printf("使用 %d 个CPU核心\n", runtime.NumCPU())
	fmt.Println("========================================")

	// 加载
	fmt.Println("\n[1/2] 加载成语数据...")
	g := LoadGraph(getIdiomFile())
	fmt.Printf("加载完成: %d 个成语\n", g.N)
	fmt.Println("\n[1.5/2] 剪枝...")
	g = PruneDeadEnds(g)

	evalPlayer := NewRandomPlayer(g)

	// 训练参数
	totalEpisodes := 5_000_000   // 500万局
	evalEvery := 100_000         // 每10万局评估
	lr := 0.05
	gamma := 0.85
	epsStart := 0.3
	numCPU := runtime.NumCPU()

	fmt.Printf("\n[2/2] 开始训练: %d 局, 每 %d 局评估一次\n", totalEpisodes, evalEvery)
	fmt.Printf("  评估点数: %d 个\n", totalEpisodes/evalEvery)

	// 初始化Q表
	qTable := make([]map[int]float64, g.N)
	for i := range g.N {
		qTable[i] = make(map[int]float64)
		for _, v := range g.Adj[i] { qTable[i][v] = rand.Float64()*0.2 - 0.1 }
	}
	nodeIDs := make([]int, g.N)
	for i := range g.N { nodeIDs[i] = i }

	// 评估函数
	evalFunc := func(qt []map[int]float64, episode int) float64 {
		evalP := &QLPlayer{qTable: qt, adj: g.Adj, nodeIDs: nodeIDs}
		wins := RunTournament(g, evalP, evalPlayer, 1000)
		return float64(wins) / 10.0
	}

	// 初始评估
	fmt.Printf("\n  %-8s | %-8s | %s\n", "累计局数", "胜率(%)", "阶段")
	fmt.Println("  ---------+----------+------------------")

	type Record struct { Episode int; WinRate float64 }
	records := []Record{}

	// 初始
	wr := evalFunc(qTable, 0)
	records = append(records, Record{0, wr})
	fmt.Printf("  %-8d | %-8.2f | %s\n", 0, wr, "随机初始")

	// 每核本地训练
	type WorkerResult struct {
		qTable []map[int]float64
		steps  int
	}

	startTime := time.Now()
	prevEp := 0

	for ep := evalEvery; ep <= totalEpisodes; ep += evalEvery {
		deltaEp := ep - prevEp
		pwDelta := deltaEp / numCPU

		results := make([]WorkerResult, numCPU)
		for w := range results {
			results[w].qTable = make([]map[int]float64, g.N)
			for i := range qTable {
				results[w].qTable[i] = make(map[int]float64)
				for v, q := range qTable[i] { results[w].qTable[i][v] = q }
			}
		}

		var wg sync.WaitGroup
		for w := 0; w < numCPU; w++ {
			wg.Add(1)
			go func(wid int) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(wid*997+ep)))
				local := &results[wid]
				used := make([]bool, g.N)
				type Move struct{ player, from, to int }
				history := make([]Move, 0, 500)

				for e := 0; e < pwDelta; e++ {
					for i := range used { used[i] = false }
					cur := nodeIDs[rng.Intn(g.N)]
					used[cur] = true
					history = history[:0]
					player := 0
					progress := float64(ep) / float64(totalEpisodes)
					eps := epsStart*(1-progress) + 0.05
					currentLR := lr * (1.0 - progress*0.9 + 0.1)
					opponentIsRandom := rng.Float64() < 0.3

					for step := 0; step < 500; step++ {
						neighbors := g.Adj[cur]
						valid := make([]int, 0, len(neighbors))
						for _, v := range neighbors { if !used[v] { valid = append(valid, v) } }
						if len(valid) == 0 {
							winner := 1 - player
							for i := len(history) - 1; i >= 0; i-- {
								m := history[i]
								var reward float64
								if i == len(history)-1 { if m.player == winner { reward = 1.0 } else { reward = -1.0 } } else { if m.player == winner { reward = 0.5 } else { reward = -0.5 } }
								oldQ := local.qTable[m.from][m.to]
								var nextMax float64
								if i < len(history)-1 { nf := history[i+1].to; for _, nv := range g.Adj[nf] { if q := local.qTable[nf][nv]; q > nextMax { nextMax = q } } }
								local.qTable[m.from][m.to] = oldQ + currentLR*(reward+gamma*(-nextMax)-oldQ)
							}
							break
						}
						var move int
						isRand := opponentIsRandom && player == 1
						if isRand { move = valid[rng.Intn(len(valid))] } else {
							if rng.Float64() < eps { move = valid[rng.Intn(len(valid))] } else {
								bm, bq := valid[0], -1e9
								for _, v := range valid { if q := local.qTable[cur][v]; q > bq { bq = q; bm = v } }
								move = bm
							}
						}
						used[move] = true
						history = append(history, Move{player, cur, move})
						cur = move; player = 1 - player
					}
					local.steps += len(history)
				}
			}(w)
		}
		wg.Wait()

		// 合并
		for i := range qTable {
			for v := range qTable[i] {
				var sum float64
				for w := 0; w < numCPU; w++ { sum += results[w].qTable[i][v] }
				qTable[i][v] = sum / float64(numCPU)
			}
		}

		totalSteps := 0
		for w := 0; w < numCPU; w++ { totalSteps += results[w].steps }

		// 评估
		wr = evalFunc(qTable, ep)
		records = append(records, Record{ep, wr})

		stage := ""
		if ep < 100000 { stage = "随机初始" } else if ep < 500000 { stage = "快速学习" } else if ep < 2000000 { stage = "稳定提升" } else { stage = "收敛期" }
		fmt.Printf("  %-8d | %-8.2f | %s (%d步)\n", ep, wr, stage, totalSteps)

		prevEp = ep
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\n训练完成: %.1f 分钟\n", elapsed.Minutes())

	// 保存结果
	type OutputRecord struct { Episode int `json:"episode"`; WinRate float64 `json:"win_rate"` }
	output := make([]OutputRecord, len(records))
	for i, r := range records { output[i] = OutputRecord{r.Episode, r.WinRate} }

	jsonData, _ := json.MarshalIndent(output, "", "  ")
	outPath := getResultPath("training_curve_dense.json")
	os.WriteFile(outPath, jsonData, 0644)
	fmt.Printf("数据保存到: %s (%d 个数据点)\n", outPath, len(output))

	// 生成图表
	genChart(outPath)
}

func genChart(dataPath string) {
	data, _ := os.ReadFile(dataPath)
	var records []struct{ Episode int; WinRate float64 }
	json.Unmarshal(data, &records)

	// 简单 ASCII 图
	fmt.Println("\n=== 训练胜率曲线 (ASCII) ===")
	maxWR := 0.0
	minWR := 100.0
	for _, r := range records {
		if r.WinRate > maxWR { maxWR = r.WinRate }
		if r.WinRate < minWR { minWR = r.WinRate }
	}
	
	for _, r := range records {
		pct := r.Episode / 10000
		bar := int((r.WinRate - minWR) / (maxWR - minWR + 0.001) * 50)
		fmt.Printf("  %4d万局 | %5.1f%% | %s\n", pct, r.WinRate, strings.Repeat("█", bar))
	}
}
