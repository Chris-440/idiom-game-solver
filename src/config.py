#!/usr/bin/env python3
"""
跨平台路径配置模块
所有路径都相对于项目根目录动态计算，确保跨平台兼容
"""

import os

# 项目根目录：src 的父目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 成语数据文件路径
IDIOM_FILE = os.path.join(DATA_DIR, 'chinese-xinhua-master', 'data', 'idiom.json')

# 结果目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# 日志目录
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 确保必要目录存在
def ensure_dirs():
    """确保输出目录存在"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

# 初始化时创建目录
ensure_dirs()


def get_idiom_file():
    """获取成语数据文件路径"""
    if not os.path.exists(IDIOM_FILE):
        # 尅试备用路径
        backup_path = os.path.join(DATA_DIR, 'idiom.json')
        if os.path.exists(backup_path):
            return backup_path
    return IDIOM_FILE


def get_result_path(filename):
    """获取结果文件路径"""
    return os.path.join(RESULTS_DIR, filename)


def get_log_path(filename):
    """获取日志文件路径"""
    return os.path.join(LOGS_DIR, filename)