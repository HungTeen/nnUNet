import os.path
from fileinput import filename
from multiprocessing import Pool
from time import time

import numpy as np

from pangteen import config, utils
from pangteen.util.resample import resample


def cut(origin_folder, origin_filename, expand_std=100, expand_var=25):
    start_time = time()

    print(f"Cutting {origin_filename} ...")
    label_filename = origin_filename.replace('volume', 'segmentation')
    case_id = int(origin_filename.split('-')[1].split('.')[0])
    image_name = "liver_{:03d}.nii.gz".format(case_id)
    label_name = "liver_{:03d}.nii.gz".format(case_id)
    # 如果存在对应文件，则跳过。
    if os.path.exists(os.path.join(config.lits_config.cut_image_folder, image_name)):
        print(f"Skip {origin_filename} ! Time cost: {time() - start_time:.2f}s")
        return

    origin_image, origin_array, origin_spacing = utils.read_image(origin_folder, origin_filename)
    label_image, label_array, label_spacing = utils.read_image(config.lits_config.label_folder, label_filename)
    print(origin_array.shape, origin_spacing)

    # 重采样。
    zSize = int(origin_array.shape[2] * origin_spacing[2])
    new_spacing = (origin_spacing[0], origin_spacing[1], 1)
    new_shape = (origin_array.shape[0], origin_array.shape[1], zSize)
    new_origin_array = resample(origin_array, new_shape, False)
    new_label_array = resample(label_array, new_shape, True)

    # 裁剪
    points = np.where(new_label_array != 0)
    min_z, max_z = new_shape[2], 0
    for x, y, z in zip(*points):
        min_z = min(min_z, z)
        max_z = max(max_z, z)

    expand = int(np.random.uniform(expand_std - expand_var, expand_std + expand_var))
    min_z = max(0, min_z - expand)
    max_z = min(new_shape[2], max_z + expand)
    print("Min z: ", min_z, "Max z: ", max_z)
    cut_origin_array = new_origin_array[:, :, min_z:max_z]
    cut_label_array = new_label_array[:, :, min_z:max_z]

    utils.save_image(cut_origin_array, origin_image, config.lits_config.cut_image_folder, image_name, spacing=new_spacing)
    utils.save_image(cut_label_array, label_image, config.lits_config.cut_label_folder, label_name, spacing=new_spacing)
    print(f"Cut {origin_filename} ! Time cost: {time() - start_time:.2f}s")


if __name__ == '__main__':
    '''
    将原始的肝脏数据集裁剪的更小一些。
    nohup python -u pangteen/ablation/cut_image.py > main.out 2>&1 &
    '''
    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start cut tasks !")
    start_time = time()
    origin_folder = config.lits_config.image_folder  # 原始数据集的标签文件夹。

    # max_zSize = 0
    # for filename in utils.next_file(origin_folder, sort=True):
    #     image, array, spacing = utils.read_image(origin_folder, filename)
    #     real_size = (array.shape[0] * spacing[0], array.shape[1] * spacing[1], array.shape[2] * spacing[2])
    #     print(filename, real_size)
    #     # 找到非空的最小包围盒。
    #     min_z, max_z = array.shape[2], 0
    #     points = np.where(array != 0)
    #     for x, y, z in zip(*points):
    #         min_z = min(min_z, z)
    #         max_z = max(max_z, z)
    #     print(min_z, max_z)
    #     max_z = max(max_z, real_size[2])
    #
    # print("Max z size: ", max_z)

    for filename in utils.next_file(origin_folder, sort=True):
        p.apply_async(cut, args=(origin_folder, filename))

    p.close()
    p.join()
    print("=========> Finish cut tasks ! Time cost: {:.2f}s".format(time() - start_time))