import os.path
from collections import OrderedDict
from multiprocessing import Pool
from time import time

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
import pandas as pd
from skimage.transform import resize

from pangteen import config, utils
from pangteen.util.resample import resample


def transform(origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename,
              target_filename, targetXYSize=256, targetZSpacing=1):
    print("Processing: ", filename, " -> ", target_filename)
    # 如果已经存在，则跳过。
    if os.path.exists(os.path.join(target_image_folder, target_filename)):
        print("Skip: ", filename)
        return

    data_image, data_array, data_spacing = utils.read_image(origin_image_folder, filename)
    label_image, label_array, label_spacing = utils.read_image(origin_label_folder, filename)
    assert data_spacing[0] == label_spacing[0] and data_spacing[1] == label_spacing[1] and data_spacing[2] == \
           label_spacing[2]
    assert data_array.shape == label_array.shape
    # Resize the data and label.
    x, y, z = data_array.shape[0] * data_spacing[0], data_array.shape[1] * data_spacing[1], data_array.shape[2] * \
              data_spacing[2]
    targetX, targetY, targetZ = targetXYSize, targetXYSize, int(z / targetZSpacing)
    targetSpacing = [x / targetX, y / targetY, z / targetZ]
    targetData = resample(data_array, [targetX, targetY, targetZ], False)
    targetLabel = resample(label_array, [targetX, targetY, targetZ], True)

    # Save the data and label.
    utils.save_image(targetData, data_image, target_image_folder, target_filename, spacing=targetSpacing)
    utils.save_image(targetLabel, label_image, target_label_folder, target_filename, spacing=targetSpacing)


def deprecated_transform():
    """
    将原始Ablation数据的尺寸和Spacing进行转换，方便适配不同的模型。
    python -u pangteen/ablation/ablation_transform.py
    nohup python -u pangteen/ablation/ablation_transform.py > main3.out 2>&1 &
    """
    origin_image_folder = config.ablation_config.origin_image_folder
    origin_label_folder = config.ablation_config.origin_label_folder
    target_image_folder = config.my_ablation_config.image_folder
    target_label_folder = config.my_ablation_config.label_folder

    case_id = 0

    rows = []
    mapping_table = []

    utils.maybe_mkdir(target_image_folder)
    utils.maybe_mkdir(target_label_folder)

    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()

    for filename in utils.next_file(origin_image_folder, sort=True):
        # Rename the file.
        target_filename = 'Ablation_{:04d}.nii.gz'.format(case_id)
        case_id += 1

        # Save Mapping.
        rows.append(filename)
        mapping_table.append([target_filename])

        p.apply_async(transform, args=(
            origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename,
            target_filename))

    pd.DataFrame(mapping_table, index=rows, columns=['new_name']).to_excel('ablation_mapping.xlsx')
    p.close()
    p.join()
    print("=========> Finish all transforms ! Cost {} seconds.".format(time() - start_time))


def transform_tumor_ablation():
    '''
    将原始Ablation数据的尺寸和Spacing进行转换，方便适配不同的模型。
    python -u pangteen/ablation/ablation_transform.py
    nohup python -u pangteen/ablation/ablation_transform.py > main3.out 2>&1 &
    '''
    origin_image_folder = config.tumor_ablation_config.rescale_image_folder
    origin_label_folder = config.tumor_ablation_config.compose_label_folder
    target_image_folder = config.tumor_ablation_config.image_folder
    target_label_folder = config.tumor_ablation_config.label_folder

    case_id = 0

    rows = []
    mapping_table = []

    utils.maybe_mkdir(target_image_folder)
    utils.maybe_mkdir(target_label_folder)

    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()

    # 先转换有标签1/2/3的消融数据集。
    for filename in utils.next_file(origin_image_folder, sort=True):
        # Rename the file.
        target_filename = 'LiverAblation_{:04d}.nii.gz'.format(case_id)
        case_id += 1

        # Save Mapping.
        rows.append(filename)
        mapping_table.append([target_filename])

        p.apply_async(transform, args=(
            origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename,
            target_filename))

    origin_image_folder = config.lits_config.cut_image_folder
    origin_label_folder = config.lits_config.cut_label_folder
    # 再转换只有标签1/2的肝脏肿瘤数据集。
    for filename in utils.next_file(origin_image_folder, sort=True):
        # Rename the file.
        target_filename = 'LiverAblation_{:04d}.nii.gz'.format(case_id)
        case_id += 1

        # Save Mapping.
        rows.append(filename)
        mapping_table.append([target_filename])

        p.apply_async(transform, args=(
            origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename,
            target_filename))

    pd.DataFrame(mapping_table, index=rows, columns=['new_name']).to_excel('ablation_mapping.xlsx')

    p.close()
    p.join()
    print("=========> Finish all transforms ! Cost {} seconds.".format(time() - start_time))


if __name__ == '__main__':
    '''
    将原始Ablation数据的尺寸和Spacing进行转换，方便适配不同的模型。
    python -u pangteen/ablation/ablation_transform.py
    nohup python -u pangteen/ablation/ablation_transform.py > main3.out 2>&1 &
    '''
    origin_image_folder = config.tumor_ablation_config.rescale_image_folder
    origin_label_folder = config.liver_ablation_config.compose_label_folder
    target_image_folder = config.liver_ablation_config.image_folder
    target_label_folder = config.liver_ablation_config.label_folder

    case_id = 0

    rows = []
    mapping_table = []

    utils.maybe_mkdir(target_image_folder)
    utils.maybe_mkdir(target_label_folder)

    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()

    # 先转换有标签1/2/3的消融数据集。
    for filename in utils.next_file(origin_image_folder, sort=True):
        # Rename the file.
        target_filename = 'LiverAblation_{:04d}.nii.gz'.format(case_id)
        case_id += 1

        # Save Mapping.
        rows.append(filename)
        mapping_table.append([target_filename])

        p.apply_async(transform, args=(
        origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename, target_filename))

    origin_image_folder = config.liver_ablation_config.history_image_folder
    origin_label_folder = config.liver_ablation_config.history_label_folder
    # 再转换只有标签1/2的肝脏肿瘤数据集。
    for filename in utils.next_file(origin_image_folder, sort=True):
        # Rename the file.
        target_filename = 'LiverAblation_{:04d}.nii.gz'.format(case_id)
        case_id += 1

        # Save Mapping.
        rows.append(filename)
        mapping_table.append([target_filename])

        p.apply_async(transform, args=(
        origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename, target_filename))

    pd.DataFrame(mapping_table, index=rows, columns=['new_name']).to_excel('ablation_mapping.xlsx')

    p.close()
    p.join()
    print("=========> Finish all transforms ! Cost {} seconds.".format(time() - start_time))
