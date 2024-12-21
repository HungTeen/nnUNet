import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import subfiles
from matplotlib import pyplot as plt
from numpy.core.defchararray import isnumeric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pangteen import config, utils
import seaborn as sns


def collect_all_image_size():
    folder = config.origin_image_folder
    table = []
    ids = []
    for filename in utils.next_file(folder):
        print(filename)
        image, array, spacing = utils.read_image_new(folder, filename)
        bb = utils.update_bounding_box((array != 0), (0, 0, 0,) + array.shape)
        shape = (bb[3] - bb[0] + 1, bb[4] - bb[1] + 1, bb[5] - bb[2] + 1)
        ids.append(filename)
        row = [shape[0] * spacing[0], shape[1] * spacing[1], shape[2] * spacing[2], np.min(array), np.max(array)]
        table.append(row)
    table = pd.DataFrame(table, index=ids, columns=['x', 'y', 'z', 'min', 'max'])
    # 保存到excel。
    table.to_excel('images.xlsx')
    print("Finish collecting all image size.")


def collect_hu():
    image_folder = config.origin_image_folder
    label_folder = config.origin_label_folder
    table = []
    bins = {}
    for filename in utils.next_file(image_folder):
        print(filename)
        if '_0000' in filename:
            # 构建新的文件名
            new_filename = filename.replace('_0000', '')
            # 获取完整的文件路径
            old_file_path = os.path.join(image_folder, filename)
            new_file_path = os.path.join(image_folder, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
        else:
            new_filename = filename

        _, image_array, _ = utils.read_image_new(image_folder, new_filename)
        _, label_array, _ = utils.read_image_new(label_folder, new_filename)
        res = image_array * (label_array != 0)
        unique, counts = np.unique(res, return_counts=True)

        # 输出结果
        for value, count in zip(unique, counts):
            bins.setdefault(value, 0)
            bins[value] += count

        bins[0] -= np.sum((label_array == 0))

    data2 = []
    for value, count in bins.items():
        print(f'值: {value}, 频数: {count}')
        data2.append([value, count])  # 保存值和频数

    data_array2 = np.array(data2)
    data_array2 = data_array2[np.argsort(data_array2[:, 0])]
    table = pd.DataFrame(data_array2, columns=['数值', '数量'])
    # 保存到excel。
    table.to_excel('images.xlsx')

    # 根据频数生成数据数组
    data = []
    for value, count in bins.items():
        data.extend([value] * count)  # 根据值的频数重复值

    data_array = np.array(data)
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    sns.histplot(data_array, bins=1000, kde=True)

    # 添加标题和标签
    plt.title('数据分布的柱状图', fontsize=16)
    plt.xlabel('值', fontsize=14)
    plt.ylabel('频率', fontsize=14)
    # 显示图形
    plt.show()

    print("Finish collecting all image size.")

def contrast(image_folder=config.origin_image_folder, result_folder=config.origin_contrast_folder, suffix = "", lower_hu = -100, upper_hu = 300):
    for filename in utils.next_file(image_folder):
        image, image_array, _ = utils.read_image_new(image_folder, filename)
        lower_mask = image_array < lower_hu
        upper_mask = image_array > upper_hu
        image_array[lower_mask] = 0
        image_array[upper_mask] = 255
        common_mask = ~lower_mask & ~upper_mask
        image_array[common_mask] = (image_array[common_mask] - lower_hu) / (upper_hu - lower_hu) * 255
        # utils.save_image_new(image_array, image, result_folder, filename)
        filename = utils.wrap_niigz(utils.unwrap_niigz(filename) + suffix)
        utils.save_image_new(image_array, image, result_folder, filename)
        print(f"Finish processing {filename}.")

    print("Finish contrast.")


def main():
    # collect_all_image_size()
    # collect_hu()
    # contrast()
    # contrast(config.origin_new_image_folder, config.origin_test_folder, "_0000")
    pass


if __name__ == '__main__':
    main()
