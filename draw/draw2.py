import matplotlib.pyplot as plt
import numpy as np

modes = ['googlenet', 'darknet19', 'vgg', 'resnet50']

data = {
    'PE Fail': [52090781, 219004425, 725936576, 561830947],
    'Router Fail': [51896668, 71611729, 113769587, 268451756],
    'Link Fail': [51888048, 70400764, 112641288, 256197952],
    'Normal': [51877393, 70224888, 110864768, 226727570],
}

ratios = [
    ['1.01x', '1.00x', '1.00x'],
    ['3.12x', '1.02x', '1.01x'],
    ['6.55x', '1.03x', '1.02x'],
    ['2.48x', '1.18x', '1.13x'],
]

bar_width = 0.2
x = np.arange(len(modes))

fig, ax = plt.subplots(figsize=(4, 3))

# 画柱状图
ax.bar(x - 2 * bar_width, data['PE Fail'], width=bar_width, label='PE Fail', color='#9467bd', hatch='//')
ax.bar(x - bar_width, data['Router Fail'], width=bar_width, label='Router Fail', color='#1f77b4', hatch='||')
ax.bar(x, data['Link Fail'], width=bar_width, label='Link Fail', color='#ff7f0e', hatch='\\\\')
ax.bar(x + bar_width, data['Normal'], width=bar_width, label='Normal', color='#2ca02c')

# 添加上方倍率标签
for i in range(len(x)):
    ax.text(x[i] - 2 * bar_width, data['PE Fail'][i]+1000000, ratios[i][0], ha='center', fontsize=8)
    ax.text(x[i] - bar_width, data['Router Fail'][i]+1000000, ratios[i][1], ha='center', fontsize=8)
    ax.text(x[i], data['Link Fail'][i]+1000000, ratios[i][2], ha='center', fontsize=8)

# ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# 设置横轴
ax.set_xticks(x)
ax.set_xticklabels(modes)
ax.set_ylabel('Inference Time (cycles)')
ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()
