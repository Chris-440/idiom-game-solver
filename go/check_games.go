package main

import (
	"fmt"
	"math/rand"
)

func main() {
	rand.Seed(42)
	
	// 真实剪枝后的图统计
	// 从之前运行结果: 29457节点, 平均出度约100, 但分布极度不均
	// 大部分节点出度很小, 少数节点出度很大
	
	// 关键问题: 游戏的实际结束条件
	// 不是 used 耗尽, 而是当前节点的所有后继都被 used
	
	// 模拟一个简化但有现实意义的模型:
	// 假设图由多个"局部集群"组成, 每个集群内部连通, 集群间连接稀疏
	// 玩家一旦进入某个小集群, 很快就会用完该集群内的节点
	
	N := 29457
	clusterSize := 50    // 平均每个集群大小
	externalLinks := 3   // 平均每个节点连向外部的边数
	
	fmt.Printf("=== 真实游戏长度分析 ===\n\n")
	fmt.Printf("总节点: %d, 假设分 %d 个集群\n", N, N/clusterSize)
	
	totalLen := 0
	for sim := 0; sim < 5000; sim++ {
		used := make(map[int]bool)
		clusterID := rand.Intn(N / clusterSize)
		
		// 在一个集群内随机游走
		length := 0
		for step := 0; step < 500; step++ {
			// 当前集群内可用节点
			availInCluster := 0
			for i := 0; i < clusterSize; i++ {
				nodeID := clusterID*clusterSize + i
				if !used[nodeID] {
					availInCluster++
				}
			}
			
			if availInCluster == 0 {
				// 尝试跳到外部
				if externalLinks == 0 || rand.Float64() < 0.5 {
					break // 无路可走
				}
				clusterID = (clusterID + rand.Intn(5) + 1) % (N / clusterSize)
				continue
			}
			
			// 选择集群内一个未使用的节点
			for {
				nodeID := clusterID*clusterSize + rand.Intn(clusterSize)
				if !used[nodeID] {
					used[nodeID] = true
					length++
					break
				}
			}
		}
		totalLen += length
	}
	
	fmt.Printf("实际平均游戏长度 ≈ %d 步\n", totalLen/5000)
	fmt.Printf("\n这意味着每局游戏只探索了总图的 %.3f%%\n", float64(totalLen/5000)/float64(N)*100)
	fmt.Printf("\n=== 关键结论 ===\n")
	fmt.Printf("游戏在局部进行 → QL只学习'常见局部路径'的Q值\n")
	fmt.Printf("全局3万节点的图 → 实际有效学习的子图只有 ~%d 个节点\n", totalLen/5000*2)
}
