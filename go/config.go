package main

import (
	"os"
	"path/filepath"
)

// 获取项目根目录
func getProjectRoot() string {
	// 优先级1: 环境变量
	if root := os.Getenv("IDIOM_PROJECT_ROOT"); root != "" {
		return root
	}

	// 优先级2: 当前工作目录
	if wd, err := os.Getwd(); err == nil {
		// 检查是否包含 go 目录（说明在项目根目录）
		goDir := filepath.Join(wd, "go")
		if _, err := os.Stat(goDir); err == nil {
			return wd
		}
		// 检查是否在 go 子目录
		if filepath.Base(wd) == "go" {
			return filepath.Dir(wd)
		}
	}

	// 优先级3: 可执行文件所在目录的父目录
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		// 如果在 go 目录，返回父目录
		if filepath.Base(exeDir) == "go" {
			return filepath.Dir(exeDir)
		}
		return exeDir
	}

	// 默认: 当前目录
	return "."
}

// 项目根目录（全局变量，启动时初始化）
var ProjectRoot string

// 初始化项目路径
func initPaths() {
	ProjectRoot = getProjectRoot()
}

// 数据目录
func getDataDir() string {
	return filepath.Join(ProjectRoot, "data")
}

// 成语数据文件
func getIdiomFile() string {
	primary := filepath.Join(getDataDir(), "chinese-xinhua-master", "data", "idiom.json")
	if _, err := os.Stat(primary); err == nil {
		return primary
	}
	// 备用路径
	backup := filepath.Join(getDataDir(), "idiom.json")
	if _, err := os.Stat(backup); err == nil {
		return backup
	}
	return primary
}

// 结果目录
func getResultsDir() string {
	dir := filepath.Join(ProjectRoot, "results")
	os.MkdirAll(dir, 0755)
	return dir
}

// 结果文件路径
func getResultPath(filename string) string {
	return filepath.Join(getResultsDir(), filename)
}

// 日志目录
func getLogsDir() string {
	dir := filepath.Join(ProjectRoot, "logs")
	os.MkdirAll(dir, 0755)
	return dir
}

// 日志文件路径
func getLogPath(filename string) string {
	return filepath.Join(getLogsDir(), filename)
}