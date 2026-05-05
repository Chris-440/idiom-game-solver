package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func main() {
	// 初始化项目路径
	initPaths()

	data, err := os.ReadFile(getIdiomFile())
	if err != nil {
		panic(err)
	}
	
	var items []map[string]interface{}
	if err := json.Unmarshal(data, &items); err != nil {
		panic(err)
	}
	
	fmt.Printf("Total items: %d\n", len(items))
	if len(items) > 0 {
		fmt.Printf("First item: %v\n", items[0])
	}
	
	count := 0
	for _, item := range items {
		word, _ := item["word"].(string)
		if len(word) == 4 {
			count++
		}
	}
	fmt.Printf("4-char idioms: %d\n", count)
}
