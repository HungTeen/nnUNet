import os
from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from nnunetv2.paths import nnUNet_raw
from pangteen import config, utils


def draw_spacing_distribution():
    # 模拟生成数据集的三维 Spacing 数据
    # 这里假设我们有 100 个样本，每个样本有三维的 Spacing 值
    folder = config.kits_config.dataset_folder
    cases = os.listdir(folder)
    spacings = []
    # for case in cases:
    #     case_folder = os.path.join(folder, case)
    #     if not os.path.exists(case_folder):
    #         continue
    #     label_image, label_array, label_spacing = utils.read_image(case_folder, "segmentation.nii.gz")
    #     print("data_spacing: ", label_spacing)
    #     spacings.append(label_spacing)

    folder = join(nnUNet_raw, 'Dataset666_KiTS2023', 'labelsTr')
    for filename in utils.next_file(folder, sort=True):
        if filename.endswith('.json'):
            continue
        case_folder = os.path.join(folder, filename)
        label_image, label_array, label_spacing = utils.read_image(folder, filename)
        print("data_spacing: ", label_spacing)
        spacings.append(label_spacing)

    num_samples = 100
    spacing_array = np.array(spacings)
    x_spacing = spacing_array[:, 0]
    y_spacing = spacing_array[:, 1]
    z_spacing = spacing_array[:, 2]

    # 创建一个三维图形对象
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    scatter = ax.scatter(x_spacing, y_spacing, z_spacing, c=z_spacing, cmap='viridis', s=50)

    # 添加颜色条
    fig.colorbar(scatter, label='Z Spacing')

    # 设置坐标轴标签
    ax.set_xlabel('X Spacing')
    ax.set_ylabel('Y Spacing')
    ax.set_zlabel('Z Spacing')

    # 设置图形标题
    ax.set_title('Spacing Distribution')

    # 显示图形
    plt.show()

    # 保存图形
    fig.savefig('spacing_distribution.png')

if __name__ == '__main__':
    """
    python -u pangteen/paper.py
    """
    draw_spacing_distribution()