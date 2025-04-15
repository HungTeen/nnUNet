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


def calculate_metrics(TP, FP, FN):
    """
    根据 TP、FP、FN 计算 Dice 系数、IOU、精确率和召回率
    :param TP: 真阳性
    :param FP: 假阳性
    :param FN: 假阴性
    :return: Dice 系数, IOU, 精确率, 召回率
    """
    dice = (2 * TP) / (2 * TP + FP + FN)
    iou = TP / (TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return dice, iou, precision, recall


def calculate_recall(dice, precision, iou):
    """
    根据 Dice、Precision、IOU 计算 Recall
    :param dice: Dice 系数
    :param precision: 精确率
    :param iou: 交并比
    :return: 召回率
    """
    recall = 1 / (1 + (1 / iou) - (1 / precision))
    return recall

def recall_from_dice_precision(dice, precision):
    denominator = 2 * precision - dice
    if denominator == 0:
        return 0.0
    recall = (dice * precision) / denominator
    return recall

def precision_from_dice_recall(dice, recall):
    denominator = 2 * recall - dice
    if denominator == 0:
        return 0.0
    precision = (dice * recall) / denominator
    return precision

def check_metrics(dice, iou, precision, recall):
    """
    检查 Dice、IOU、Precision 和 Recall 指标的合理性
    :param dice: Dice 系数
    :param iou: 交并比
    :param precision: 精确率
    :param recall: 召回率
    :return: 检查结果信息
    """
    # 检查每个指标是否在 [0, 1] 区间内
    metrics = {
        'Dice': dice,
        'IOU': iou,
        'Precision': precision,
        'Recall': recall
    }
    for metric_name, metric_value in metrics.items():
        if metric_value < 0 or metric_value > 1:
            return f"{metric_name} 的值 {metric_value} 不在 [0, 1] 区间内，可能有误。"

    # 检查指标之间的关系
    # 根据公式 Dice = 2 * IOU / (1 + IOU) 进行粗略检查
    calculated_dice = 2 * iou / (1 + iou)
    if abs(dice - calculated_dice) > 0.1:  # 允许一定的误差范围
        return f"Dice 系数 {dice} 与根据 IOU 计算得到的近似值 {calculated_dice} 差异较大，可能有误。"

    # 根据公式 Precision = 1 / (1 + (1 / IOU) - (1 / Recall)) 进行检查
    calculated_precision = 1 / (1 + (1 / iou) - (1 / recall))
    if abs(precision - calculated_precision) > 0.1:  # 允许一定的误差范围
        return f"Precision {precision} 与根据 IOU 和 Recall 计算得到的近似值 {calculated_precision} 差异较大，可能有误。"

    return "所有指标看起来合理。"



if __name__ == '__main__':
    # plot()
    # plot_radar()
    # plot_comparison_bar(data = {
    #         '指标': ['肝脏', '肿瘤', '平均'],
    #         'B': [0.917, 0.579, 0.748],
    #         'B+M': [0.92, 0.67, 0.795],
    #         'B+M+I': [0.927, 0.701, 0.814],
    #         'B+M+I+K': [0.941, 0.75, 0.846],
    #         'B+M+I+K+C': [0.936, 0.777, 0.857],
    # }, title='模块消融实验结果', png_name='ablation_module.png')
    # plot_comparison_bar(data={
    #     '指标': ['肝脏', '肿瘤', '平均'],
    #     'Layer 1': [0.93, 0.738, 0.834],
    #     'Layer 2': [0.936, 0.777, 0.857],
    #     'Layer 3': [0.937, 0.766, 0.852],
    #     'Layer 4': [0.932, 0.758, 0.845],
    # }, title='Shifted KAN 模块消融', png_name='ablation_layer.png', detail=True)

    # 示例使用
    # TP = 70
    # FP = 20
    # FN = 10
    #
    # dice, iou, precision, recall = calculate_metrics(TP, FP, FN)
    # print(f"Dice 系数: {dice:.4f}")
    # print(f"IOU: {iou:.4f}")
    # print(f"精确率: {precision:.4f}")
    # print(f"召回率: {recall:.4f}")

    # 示例使用
    dice = 0.777
    iou = 0.646
    precision = 0.797
    recall = 0.795

    result = check_metrics(dice, iou, precision, recall)
    print(result)

    precision_dice = precision_from_dice_recall(dice, recall)
    print(f"Precision from Dice and Recall: {precision_dice:.4f}")

    pass