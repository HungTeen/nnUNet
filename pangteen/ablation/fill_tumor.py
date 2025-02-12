import os.path
from fileinput import filename
from multiprocessing import Pool
from time import time

import numpy as np

from pangteen import config, utils

def transform_tumor(predict_folder, predict_filename):
    """
    将历史消融区域标签与当前消融区域标签合并。
    """
    start_time = time()
    print(f"Transforming {filename} ...")
    # 读取预测的肿瘤图像。
    predict_image, predict_array, _ = utils.read_image(predict_folder, predict_filename)
    # 读取原始消融区域标注。
    label_image, label_array, _ = utils.read_image(config.tumor_ablation_config.rescale_label_folder, predict_filename)
    # 将肿瘤标签变为 2，消融区域为 1。
    predict_array[label_array == 1] = 2
    # 通过BFS，将预测错误的肿瘤区域变为空气。
    ablation_points = []
    for i in range(predict_array.shape[0]):
        for j in range(predict_array.shape[1]):
            for k in range(predict_array.shape[2]):
                if label_array[i, j, k] == 1:
                    ablation_points.append((i, j, k))

    for i, j, k in ablation_points:
        if predict_array[i, j, k] == 2:
            predict_array[i, j, k] = 0
        elif predict_array[i, j, k] == 0:
            predict_array[i, j, k] = 1
        # 遍历6个方向，如果周围的点是肿瘤，加入循环。
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            x, y, z = i + dx, j + dy, k + dz
            if 0 <= x < predict_array.shape[0] and 0 <= y < predict_array.shape[1] and 0 <= z < predict_array.shape[2]:
                if predict_array[x, y, z] == 2:
                    ablation_points.append((x, y, z))

    # 保存结果。
    utils.save_image(predict_array, predict_image, config.tumor_ablation_config.compose_label_folder, predict_filename)
    print(f"Transformed {filename} ! Time cost: {time() - start_time:.2f}s")


def transform_liver_tumor(predict_folder, predict_filename):
    """
    将肝脏+肿瘤的预测结果转换为肝脏+肿瘤+消融区域的预测结果。
    """
    start_time = time()
    print(f"Transforming {filename} ...")
    # 读取预测的肿瘤图像。
    predict_image, predict_array, _ = utils.read_image(predict_folder, predict_filename)
    # 读取原始消融区域标注。
    label_image, label_array, _ = utils.read_image(config.tumor_ablation_config.rescale_label_folder, predict_filename)
    # 将消融区域标签从 1 变为 3。
    predict_array[label_array == 1] = 3

    # 让和消融区域临近的肿瘤区域变成消融区域。
    ablation_points = np.where(label_array == 1)
    count = 0
    q = []
    for i, j, k in zip(*ablation_points):
        q.append((i, j, k))
    while len(q) > 0:
        i, j, k = q.pop(0)
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            x, y, z = i + dx, j + dy, k + dz
            if 0 <= x < predict_array.shape[0] and 0 <= y < predict_array.shape[1] and 0 <= z < predict_array.shape[2]:
                if predict_array[x, y, z] == 2:
                    predict_array[x, y, z] = 3
                    q.append((x, y, z))
                    count += 1

    print("Changed", count, "points to ablation area.")

    # 保存结果。
    utils.save_image(predict_array, predict_image, config.tumor_ablation_config.compose_label_folder, predict_filename)
    print(f"Transformed {filename} ! Time cost: {time() - start_time:.2f}s")


def transform_liver_ablation(predict_folder, predict_filename):
    """
    将LITS预测的肿瘤区域转换为消融区域。
    """
    start_time = time()
    print(f"Transforming {predict_filename} ...")
    # 读取预测的肿瘤图像。
    predict_image, predict_array, _ = utils.read_image(predict_folder, predict_filename)
    # 读取原始消融区域标注。
    label_image, label_array, _ = utils.read_image(config.tumor_ablation_config.rescale_label_folder, predict_filename)

    # 预先记录下所有的消融区域。
    ablation_points = np.where(label_array == 1)
    q = []


    # 将预测的肿瘤区域变为历史消融区域。
    label_array[predict_array == 2] = 2

    # 让和消融区域临近的肿瘤区域变成消融区域。
    for i, j, k in zip(*ablation_points):
        q.append((i, j, k))
        label_array[i, j, k] = 1

    count = 0
    while len(q) > 0:
        i, j, k = q.pop(0)
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            x, y, z = i + dx, j + dy, k + dz
            if 0 <= x < label_array.shape[0] and 0 <= y < label_array.shape[1] and 0 <= z < label_array.shape[2]:
                if label_array[x, y, z] == 2:
                    label_array[x, y, z] = 1
                    q.append((x, y, z))
                    count += 1

    print("Changed", count, "points to ablation area.")

    # 保存结果。
    utils.save_image(label_array, label_image, config.liver_ablation_config.compose_label_folder, predict_filename)
    print(f"Transformed {filename} ! Time cost: {time() - start_time:.2f}s")


if __name__ == '__main__':
    '''
    python pangteen/image_info.py Ablation/origin/data
    '''
    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()
    # folder = config.tumor_ablation_config.predict_folder
    folder = config.tumor_ablation_config.predict_folder

    for filename in utils.next_file(folder, sort=True):
        p.apply_async(transform_liver_ablation, args=(folder, filename))

    p.close()
    p.join()
    print("All done ! Time cost: ", time() - start_time)
