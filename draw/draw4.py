import matplotlib.pyplot as plt
import numpy as np

# 横轴类别
modes = ['5', '10', '15', '20']
x = np.arange(len(modes))

# 两组柱状图数据（左轴）
bar_data_1 = [0.7523, 0.7563, 0.7291, 0.7087]  # accuracy

# 折线图数据（右轴）
line_data = [225, 224, 224, 223]  # Utilization (%)

bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(5.5, 3))

# 绘制双柱（左轴）
bar1 = ax1.bar(x, bar_data_1, width=bar_width, label='Detection Accuracy', color='#2ca02c')

ax1.set_ylabel('Accuracy (%)', color='black')
ax1.set_ylim(0.7, 0.8) 
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(modes)

# 加网格
ax1.set_axisbelow(True)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# 折线图（右轴）
ax2 = ax1.twinx()
line = ax2.plot(x, line_data, color='#ff7f0e', marker='o', linestyle='--', linewidth=2, label='Time')
ax2.set_ylabel('Memory Cost (MB)', color='black')
ax2.set_ylim(150, 350) 
ax2.tick_params(axis='y', labelcolor='black')

# 图例（组合柱状图 + 折线图）
lines = [bar1[0], line[0]]
labels = ['Accuracy', 'Memory Cost']
ax1.legend(lines, labels, loc='upper left', fontsize=8)

ax1.set_xlabel('Threshold', color='black')

plt.tight_layout()
plt.show()
