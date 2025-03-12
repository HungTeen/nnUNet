import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 若使用 Linux 系统，可尝试 'WenQuanYi Zen Hei'


def plot_module_ablation_bar():
    data = {
            '指标': ['肝脏', '肿瘤', '平均'],
            'B': [0.918, 0.536, 0.727],
            'B+M': [0.92, 0.61, 0.765],
            'B+M+I': [0.925, 0.64, 0.783],
            'B+M+I+K': [0.94, 0.674, 0.807],
            'B+M+I+K+C': [0.949, 0.685, 0.817],
            'B+M+I+K+C+L': [0.948, 0.726, 0.837],
    }

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
    plt.title('模块消融实验结果', fontsize=10, pad=20)
    # plt.xlabel('Model Variants', fontsize=12)
    plt.ylabel('Dice', fontsize=10)
    plt.ylim(0.5, 1.0)

    # 添加数据标签
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=7)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('ablation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot():
    # 定义数据
    data = {
        '指标': ['Dice', 'IOU', 'HD95', 'Precision', 'Recall', 'ASSD'],
        'liver': [0.95, 0.907, 14.125, 0.946, 0.957, 3.346],
        'tumor': [0.659, 0.54, 40.939, 0.726, 0.679, 8.867],
        '平均': [0.804, 0.724, 27.532, 0.836, 0.818, 6.107],
        '最佳': [0.985, 0.972, 1.225, 0.994, 0.977, 0.575],
        '最差': [0.491, 0.48, 115.954, 0.492, 0.492, 20.275],
        '75分位': [0.893, 0.814, 5.012, 0.913, 0.924, 2.199],
        '25分位': [0.744, 0.641, 39.714, 0.783, 0.766, 7.293]
    }

    # 创建 DataFrame
    df = pd.DataFrame(data)
    # 将 '指标' 列设置为索引
    df.set_index('指标', inplace=True)

    # 绘制肝脏和肿瘤各指标得分柱状图
    bar_df = df[['liver', 'tumor']]
    bar_df.plot(kind='bar', figsize=(10, 6))
    plt.title('肝脏和肿瘤各指标得分对比')
    plt.xlabel('指标')
    plt.ylabel('得分')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制各项指标的统计信息箱线图
    box_df = df.drop(['liver', 'tumor'], axis=1)
    box_df = box_df.T
    box_df.plot(kind='box', figsize=(10, 6))
    plt.title('各项指标的统计信息箱线图')
    plt.xlabel('指标')
    plt.ylabel('得分')
    plt.xticks(rotation=45)
    plt.tight_layout()
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
    plot_module_ablation_bar()
    pass