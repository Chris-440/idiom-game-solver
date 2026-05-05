#!/usr/bin/env python3
"""
基于真实训练数据绘制训练曲线（简洁版，无多余标注）
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_result_path

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# 加载真实数据
with open(get_result_path('training_curve_dense.json')) as f:
    data = json.load(f)

episodes = [d['episode'] for d in data]
win_rates = [d['win_rate'] for d in data]

# 保存数据
full_data = {
    'episodes': episodes,
    'win_rates': win_rates,
}
with open(get_result_path('training_curve_real.json'), 'w') as f:
    json.dump(full_data, f, indent=2)

# ============================================================
# 绘制图表
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [2.5, 1]})

ep_m = [e / 10000 for e in episodes]

# ===== 上图：完整胜率曲线 =====
ax1.plot(ep_m, win_rates, 'b-o', linewidth=2.5, markersize=4, alpha=0.85, zorder=5)

# 参考线
ax1.axhline(y=50, color='#999', linestyle=':', lw=1.5, alpha=0.5)

ax1.set_xlabel('训练局数（万局）', fontsize=14, fontweight='bold')
ax1.set_ylabel('对 Random 胜率 (%)', fontsize=14, fontweight='bold')
ax1.set_title('Q-Learning 训练胜率曲线', fontsize=15, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.2, linestyle='--', lw=0.8)
ax1.set_xlim(-2, 52)
ax1.set_ylim(40, 100)

# ===== 下图：50-500万局放大 =====
ax2.plot(ep_m, win_rates, 'b-o', linewidth=2.5, markersize=5, alpha=0.85, zorder=5)

ax2.set_xlabel('训练局数（万局）', fontsize=14, fontweight='bold')
ax2.set_ylabel('胜率 (%)', fontsize=14, fontweight='bold')
ax2.set_title('局部放大（50-500万局）', fontsize=15, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.2, linestyle='--', lw=0.8)
ax2.set_xlim(45, 55)
ax2.set_ylim(88, 100)

plt.tight_layout(pad=2.0)

for fmt in ['png', 'pdf']:
    path = get_result_path(f'training_curve_real.{fmt}')
    plt.savefig(path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')
    print(f"已保存: {path}")

print(f"\n数据点数: {len(episodes)} 个")
print(f"初始: {win_rates[0]}%")
print(f"10万局: {win_rates[1]}%")
print(f"500万局: {win_rates[-1]}%")
