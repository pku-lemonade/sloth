import numpy as np
import matplotlib.pyplot as plt

apps = ['mesh4-4', 'mesh6-6', 'mesh8-8']
x = np.arange(len(apps))

a = [145556537,	132822583,	50285002]
b = [113134417,	111335378,	48209785]
c = [94167274,	99068719,	42437393]
d = [85436870,	93207976,	40497365]

width = 0.15

fig, ax = plt.subplots(figsize=(10, 5))

# ax.bar(x - width, Gemini, width, label='Gemini', color='white', edgecolor='black', hatch='//')
# ax.bar(x, OurSimulator, width, label='Our Simulator', color='black')

ax.bar(x - 2*width, a, width, label='window size = 5', color='white', edgecolor='black', hatch='//')
ax.bar(x - width, b, width, label='window size = 10', color='lightgray', edgecolor='black', hatch='\\\\')
ax.bar(x, c, width, label='window size = 15', color='darkgray', edgecolor='black', hatch='xx')
ax.bar(x + width, d, width, label='window size = 20', color='black')

ax.set_ylabel('Simulation Cycles')
ax.set_xticks(x)
ax.set_xticklabels(apps, rotation=45, ha='right')

ax.legend()

plt.tight_layout()
plt.show()