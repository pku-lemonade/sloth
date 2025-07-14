import matplotlib.pyplot as plt
import numpy as np

# 横轴类别
modes = ['2', '4', '8', '16', '32']
x = np.arange(len(modes))

# 两组柱状图数据（左轴）
bar_data_1 = [0.36, 0.71, 1.42, 2.85, 5.7]  # 计算trace
bar_data_2 = [1.93, 3.86, 7.72, 15.44, 30.88]  # 通信trace

# 折线图数据（右轴）
line_data = [189268331, 258881107, 399266547, 679663694, 1249327388]  # Utilization (%)

bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(5.5, 3))

# 绘制双柱（左轴）
bar1 = ax1.bar(x - bar_width/2, bar_data_1, width=bar_width, label='Compute Trace', color='#2ca02c')
bar2 = ax1.bar(x + bar_width/2, bar_data_2, width=bar_width, label='Communication Trace', color='#1f77b4')

ax1.set_ylabel('Memory (GB)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(modes)

# 加网格
ax1.set_axisbelow(True)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# 折线图（右轴）
ax2 = ax1.twinx()
line = ax2.plot(x, line_data, color='#ff7f0e', marker='o', linestyle='--', linewidth=2, label='Time')
ax2.set_ylabel('Time (cycles)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 图例（组合柱状图 + 折线图）
lines = [bar1[0], bar2[0], line[0]]
labels = ['Compute Trace', 'Communication Trace', 'Compute Time']
ax1.legend(lines, labels, loc='upper left', fontsize=8)

ax1.set_xlabel('Number of Inferences', color='black')

plt.tight_layout()
plt.show()
