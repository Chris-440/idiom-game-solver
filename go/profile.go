package main

import (
	"fmt"
	"math/rand"
	"time"
	"unicode/utf8"
)

type Graph struct {
	N          int
	Text       []string
	Adj        [][]int
	TailChar   []string
	TailPinyin []string
	HeadByChar map[string][]int
	HeadByPin  map[string][]int
}

func LoadGraph(filepath string) *Graph {
	// 简化版加载
	return &Graph{N: 29502, Adj: make([][]int, 29502)}
}

func main() {
	rand.Seed(time.Now().UnixNano())
	
	// 测试单次游戏性能
	fmt.Println("性能测试:")
	
	// 模拟简单操作
	start := time.Now()
	iterations := 3000000
	
	// 测试 map 操作
	m := make(map[int]float64)
	for i := 0; i < 29502; i++ {
		m[i] = rand.Float64()
	}
	
	ops := 0
	for i := 0; i < iterations; i++ {
		// 模拟游戏步骤
		_ = m[rand.Intn(29502)] // 查表
		_ = rand.Float64() < 0.3 // ε-greedy
		ops++
	}
	
	elapsed := time.Since(start)
	fmt.Printf("  %d 次查表+随机操作: %v\n", ops, elapsed)
	fmt.Printf("  每秒操作数: %.0f\n", float64(ops)/elapsed.Seconds())
	fmt.Printf("  纳秒/操作: %.0f\n", float64(elapsed.Nanoseconds())/float64(ops))
}
