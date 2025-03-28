import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle

def draw_grid(L, H, data, links):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(0, H)
    ax.set_ylim(0, L)
    ax.invert_yaxis()  # 矩阵坐标系（左上角为原点）
    ax.set_xticks([])  # 移除x轴刻度
    ax.set_yticks([])  # 移除y轴刻度
    ax.axis('off')  # 移除坐标轴

    # 绘制数字（正方形）
    square_size = 0.6  # 正方形大小
    square_padding = 0.2  # 正方形padding，类比机器学习
    for i in range(L):
        for j in range(H):
            ax.add_patch(Rectangle(
                (j + square_padding, i + square_padding), square_size, square_size,
                facecolor='white', edgecolor='black', linewidth=1
            ))
            ax.text(j + 0.5, i + 0.5, str(data[i][j]),
                    ha='center', va='center', fontsize=12)

    # 归一化颜色映射
    link_values = [link[2] for link in links]
    norm = Normalize(vmin=min(link_values), vmax=max(link_values))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['darkblue', 'blue', 'green', 'yellow', 'red'])

    # 绘制连接（长方形）
    for (i1, j1), (i2, j2), value in links:
        # 计算连接方向
        if i1 == i2 and j1 == j2-1:  # 水平连接
            x = j1 + 0.5 + 0.5*square_size
            y = i1 + 0.5 +0.08
            width = 2*square_padding
            height = 0.1  # 长方形高度更小
        elif i1 == i2 and j2 == j1-1:  # 水平连接
            x = j2 + 0.5 + 0.5*square_size
            y = i1 + 0.5 -0.08
            width = 2*square_padding
            height = 0.1  # 长方形高度更小
        elif j1 == j2 and i1==i2-1:  # 垂直连接
            x = j1 + 0.5 +0.08  # 调整位置避免重合
            y = i1 + 0.5 + 0.5*square_size
            width = 0.1  # 长方形宽度更小
            height = 2*square_padding
        elif j1 == j2 and i2==i1-1:  # 垂直连接
            x = j1 + 0.5 -0.08  # 调整位置避免重合
            y = i2 + 0.5 + 0.5*square_size
            width = 0.1  # 长方形宽度更小
            height = 2*square_padding
        else:
            continue  # 忽略斜向连接

        ax.add_patch(Rectangle(
            (x, y), width, height,
            facecolor=cmap(norm(value)),
            edgecolor='none',
            alpha=0.7
        ))

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Link Value')

    plt.show()

# 示例输入（根据你的截图数据）
if __name__ == '__main__':
    L, H = 6, 6
    data = [
        [3, 1, 1, 2, 3, 1],
        [3, 3, 2, 3, 1, 1],
        [1, 1, 3, 3, 3, 1],
        [3, 3, 3, 1, 3, 1],
        [1, 3, 2, 1, 1, 3],
        [1, 3, 2, 1, 1, 3]
    ]

    # 生成链接数据示例（需根据实际需求定义）
    links = [
        # 格式：((i1,j1), (i2,j2), value)
        ((0,0), (0,1), 2),  # 水平连接示例
        ((1,2), (2,2), 3),  # 垂直连接示例
        ((3,3), (3,4), 1),
        ((4,5), (5,5), 4),
        ((5,5), (4,5), 6.3)
    ]

    draw_grid(L, H, data, links)