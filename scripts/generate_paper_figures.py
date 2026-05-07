#!/usr/bin/env python3
"""
生成论文图表
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import os

# 设置输出目录
OUTPUT_DIR = '/Users/dzj/code/成语接龙/docs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_training_data():
    """加载Q-Learning训练数据"""
    with open('/Users/dzj/code/成语接龙/results/training_curve_real.json', 'r') as f:
        data = json.load(f)
    return data['episodes'], data['win_rates']

def load_competition_data():
    """加载对战结果数据"""
    with open('/Users/dzj/code/成语接龙/results/competition_results_go.json', 'r') as f:
        data = json.load(f)
    return data

def figure1_training_curve():
    """图1: Q-Learning训练胜率曲线"""
    episodes, win_rates = load_training_data()
    
    # 转换为万局单位
    episodes_wan = [e / 10000 for e in episodes]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制散点
    plt.scatter(episodes_wan, win_rates, c='steelblue', s=30, alpha=0.7, label='评估点')
    
    # 绘制平滑曲线（使用移动平均）
    window = 5
    smoothed = np.convolve(win_rates, np.ones(window)/window, mode='valid')
    smoothed_episodes = episodes_wan[window-1:]
    plt.plot(smoothed_episodes, smoothed, 'b-', linewidth=2, label='平滑曲线')
    
    # 标注关键阶段
    plt.axhline(y=90, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=95, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # 标注阶段区间
    plt.annotate('快速学习期', xy=(5, 88), fontsize=10, color='orange')
    plt.annotate('稳定提升期', xy=(50, 92), fontsize=10, color='gray')
    plt.annotate('收敛期', xy=(300, 96.5), fontsize=10, color='green')
    
    plt.xlabel('训练局数（万局）', fontsize=12)
    plt.ylabel('对Random胜率 (%)', fontsize=12)
    plt.title('Q-Learning训练胜率曲线', fontsize=14)
    plt.xlim(0, 500)
    plt.ylim(45, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_training_curve.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig1_training_curve.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"图1已保存: {OUTPUT_DIR}/fig1_training_curve.png")
    
    # 检查数据
    print(f"  数据点数: {len(episodes)}")
    print(f"  最大胜率: {max(win_rates)}%")
    print(f"  最小胜率: {min(win_rates)}%")
    print(f"  收敛期均值(100万-500万): {np.mean(win_rates[10:]):.2f}%")

def figure2_ppo_versions():
    """图2: PPO自博弈版本演进对比"""
    # 根据文档数据绘制
    versions = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    peak_rates = [2, 19, 76, 80, 81, 85]
    convergence_speed = [50, 1300, 300, 100, 50, 750]  # 轮数
    colors = ['red', 'red', 'green', 'blue', 'blue', 'darkgreen']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：峰值胜率
    bars1 = ax1.bar(versions, peak_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('vs Q-table峰值胜率 (%)', fontsize=12)
    ax1.set_xlabel('版本', fontsize=12)
    ax1.set_title('各版本峰值性能', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=85.6, color='gray', linestyle='--', label='理论上限(85.6%)')
    ax1.legend(fontsize=9)
    
    # 标注数值
    for bar, val in zip(bars1, peak_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val}%', ha='center', fontsize=10)
    
    # 右图：收敛速度（轮数）
    bars2 = ax2.bar(versions, convergence_speed, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('收敛轮数', fontsize=12)
    ax2.set_xlabel('版本', fontsize=12)
    ax2.set_title('各版本收敛速度', fontsize=14)
    
    # 标注数值
    for bar, val in zip(bars2, convergence_speed):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                str(val), ha='center', fontsize=10)
    
    # 标注失败版本
    ax1.annotate('失败', xy=(0, 5), fontsize=8, color='red')
    ax1.annotate('失败', xy=(1, 22), fontsize=8, color='red')
    ax1.annotate('转折', xy=(2, 79), fontsize=8, color='green')
    ax1.annotate('最优', xy=(5, 88), fontsize=8, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_ppo_versions.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig2_ppo_versions.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"图2已保存: {OUTPUT_DIR}/fig2_ppo_versions.png")

def figure3_position_analysis():
    """图3: 起始位置分类分析"""
    # 数据来自文档
    categories = ['模型先手后手均胜', '模型仅先手胜', '模型仅后手胜', '模型先手后手均负']
    counts = [1394, 455, 151, 0]
    percentages = [69.7, 22.8, 7.5, 0]
    colors = ['green', 'orange', 'blue', 'gray']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：数量分布
    bars = ax1.bar(categories[:3], counts[:3], color=colors[:3], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('位置数量', fontsize=12)
    ax1.set_title('起始位置分类数量（PPO vs Q-table）', fontsize=14)
    
    for bar, cnt, pct in zip(bars, counts[:3], percentages[:3]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{cnt}\n({pct}%)', ha='center', fontsize=10)
    
    # 右图：饼图
    nonzero_categories = ['PPO P0/P1均胜', 'PPO仅P0胜', 'PPO仅P1胜']
    nonzero_counts = [1394, 455, 151]
    nonzero_colors = ['green', 'orange', 'blue']
    
    wedges, texts, autotexts = ax2.pie(nonzero_counts, labels=nonzero_categories, 
                                        colors=nonzero_colors, autopct='%1.1f%%',
                                        startangle=90, explode=[0.02, 0.02, 0.02])
    ax2.set_title('位置类型占比', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_position_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig3_position_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"图3已保存: {OUTPUT_DIR}/fig3_position_analysis.png")

def figure4_win_rate_comparison():
    """图4: 方法对比汇总"""
    methods = ['Random', 'Q-Learning', 'PPO (V6)', '理论上限']
    vs_random = [50, 97.4, 99.8, 100]  # PPO对random约99.8%（从文档）
    vs_qtable = [0.15, 50, 85, 85.6]  # Random对Q-table约0.15%（从文档数据推算）
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, vs_random, width, label='vs Random', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, vs_qtable, width, label='vs Q-table', color='coral', alpha=0.7)
    
    ax.set_ylabel('胜率 (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_title('各方法胜率对比', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='Q-table理论上限')
    
    # 标注数值
    for bar, val in zip(bars1, vs_random):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', fontsize=9)
    for bar, val in zip(bars2, vs_qtable):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig4_method_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"图4已保存: {OUTPUT_DIR}/fig4_method_comparison.png")

def figure5_game_theory_verification():
    """图5: 游戏理论上限验证"""
    # Q-table vs Q-table 自对弈数据
    position_types = ['Only P0\n(先手必胜)', 'Only P1\n(后手反杀)']
    p0_win_rate = [93.8, 8.6]
    p1_win_rate = [6.2, 91.4]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(position_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, p0_win_rate, width, label='P0胜率', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, p1_win_rate, width, label='P1胜率', color='coral', alpha=0.7)
    
    ax.set_ylabel('胜率 (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(position_types, fontsize=11)
    ax.set_title('Q-table自对弈：验证先手优势为游戏本质', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    
    # 标注结论
    ax.annotate('先手必胜位置\nP0优势明显', xy=(-0.175, 50), fontsize=9, ha='center')
    ax.annotate('后手反杀位置\nP1优势明显', xy=(0.825, 50), fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_theory_verification.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig5_theory_verification.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"图5已保存: {OUTPUT_DIR}/fig5_theory_verification.png")

def main():
    print("=" * 50)
    print("生成论文图表")
    print("=" * 50)
    print()
    
    figure1_training_curve()
    print()
    
    figure2_ppo_versions()
    print()
    
    figure3_position_analysis()
    print()
    
    figure4_win_rate_comparison()
    print()
    
    figure5_game_theory_verification()
    print()
    
    print("=" * 50)
    print("所有图表生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == '__main__':
    main()