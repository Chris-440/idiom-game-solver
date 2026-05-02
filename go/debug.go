package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func main() {
	data, err := os.ReadFile("/Users/dzj/code/成语接龙/data/chinese-xinhua-master/data/idiom.json")
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
