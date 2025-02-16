import os.path
from fileinput import filename
from multiprocessing import Pool
from time import time

import numpy as np
from scipy.ndimage import uniform_filter

from pangteen import config, utils
from pangteen.util import excels


def fill_hole(origin_folder, origin_filename):
    """
    将LITS预测的肿瘤区域转换为消融区域。
    """
    start_time = time()
    print(f"Transforming {origin_filename} ...")
    # 读取预测的肿瘤图像。
    origin_image, origin_array, _ = utils.read_image(origin_folder, origin_filename)
    # 读取肝脏预测的标签。
    # mapping_name = excels.get_mapped_name(origin_filename, config.tumor_ablation_config.base_folder)
    # print("Mapping Name: ", mapping_name)
    # liver_image, liver_array, _ = utils.read_image(config.tumor_ablation_config.predict_folder, mapping_name)
    # 读取原始消融区域标注。
    label_image, label_array, _ = utils.read_image(config.ablation_config.label_folder, origin_filename)
    # 采用大小为3*3*3的平滑核对肝脏预测的标签进行平滑。
    # new_array = np.zeros_like(origin_array)
    kernel_size = 3
    # radius = kernel_size // 2
    # for i in range(radius, origin_array.shape[0] - radius):
    #     for j in range(radius, origin_array.shape[1] - radius):
    #         for k in range(radius, origin_array.shape[2] - radius):
    #             mean_hu = np.mean(origin_array[i - radius:i + radius + 1, j - radius:j + radius + 1, k - radius:k + radius + 1])
    #             new_array[i, j, k] = mean_hu
    kernel_size = 3  # 3x3x3 立方体
    new_array = origin_array
    for i in range(3):
        new_array = uniform_filter(new_array, size=kernel_size, mode='constant')

    utils.save_image(new_array, origin_image, config.predict_folder, origin_filename)
    print("Time cost: ", time() - start_time)


if __name__ == '__main__':
    '''
    python pangteen/image_info.py Ablation/origin/data
    '''
    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()
    folder = config.ablation_config.image_folder

    for filename in utils.next_file(folder, sort=True):
        fill_hole(folder, filename)
        # p.apply_async(fill_hole, args=(folder, filename), error_callback=utils.print_error)
        break

    p.close()
    p.join()
    print("All done ! Time cost: ", time() - start_time)