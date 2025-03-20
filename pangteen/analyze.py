import os.path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 若使用 Linux 系统，可尝试 'WenQuanYi Zen Hei'


def plot_comparison_bar(data: dict, title: str, png_name: str, detail=False):
    df = pd.DataFrame(data).set_index('指标')
    bar_df = df.T

    plt.figure(figsize=(12, 6), dpi=300)
    ax = bar_df.plot(kind='bar',
                     width=0.85,  # 1.扩大柱子宽度（原为0.8）
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                     edgecolor='black')

    # 2.精确居中标签的关键配置
    n_models = len(bar_df.index)  # 模型数量
    n_bars = len(bar_df.columns)  # 每个模型的柱子数（3）

    # 计算柱群中心位置
    xticks_pos = [i + (n_bars - 3) / 2 * 0.85 / n_bars for i in range(n_models)]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(bar_df.index,
                       rotation=0,
                       ha='center')  # 重要：强制居中对齐

    # 其余配置（保持不变）
    plt.title(title, fontsize=10, pad=20)
    # plt.xlabel('Model Variants', fontsize=12)
    plt.ylabel('Dice', fontsize=10)
    plt.ylim(0.5, 1.0)

    # 添加数据标签
    for p in ax.patches:
        if detail:
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=7)
        else:
            ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=7)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.abspath('E:\Study\研究生\矢量图')
    plt.savefig(os.path.join(save_path, png_name), dpi=300, bbox_inches='tight')
    plt.show()


def plot_radar():
    import numpy as np
    import matplotlib.pyplot as plt

    # 示例数据
    labels = ['指标 1', '指标 2', '指标 3', '指标 4', '指标 5', '指标 6', '指标 7']
    models = ['模型 1', '模型 2', '模型 3', '模型 4', '模型 5', '模型 6']
    data = np.random.rand(6, 7)  # 随机生成数据

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        values = data[i].tolist()
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('模型指标对比')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

if __name__ == '__main__':
    # plot()
    # plot_radar()
    plot_comparison_bar(data = {
            '指标': ['肝脏', '肿瘤', '平均'],
            'B': [0.917, 0.579, 0.748],
            'B+M': [0.92, 0.67, 0.795],
            'B+M+I': [0.927, 0.701, 0.814],
            'B+M+I+K': [0.941, 0.75, 0.846],
            'B+M+I+K+C': [0.936, 0.777, 0.857],
    }, title='模块消融实验结果', png_name='ablation_module.png')
    plot_comparison_bar(data={
        '指标': ['肝脏', '肿瘤', '平均'],
        'Layer 1': [0.93, 0.738, 0.834],
        'Layer 2': [0.936, 0.777, 0.857],
        'Layer 3': [0.937, 0.766, 0.852],
        'Layer 4': [0.932, 0.758, 0.845],
    }, title='Shifted KAN 模块消融', png_name='ablation_layer.png', detail=True)
    pass