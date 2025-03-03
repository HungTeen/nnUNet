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


def transform(dataset_folder, case_folder_name, targetXSpacing=1):
    print("Processing: ", case_folder_name)
    case_folder = os.path.join(dataset_folder, case_folder_name)
    filename = utils.wrap_niigz(case_folder_name)
    # 如果已经存在，则跳过。
    if not os.path.exists(case_folder):
        print("Skip: ", case_folder)
        return

    data_image, data_array, data_spacing = utils.read_image(case_folder, "imaging.nii.gz")
    label_image, label_array, label_spacing = utils.read_image(case_folder, "segmentation.nii.gz")
    assert data_spacing[0] == label_spacing[0] and data_spacing[1] == label_spacing[1] and data_spacing[2] == \
           label_spacing[2]
    assert data_array.shape == label_array.shape
    # Resize the data and label.
    x, y, z = data_array.shape[0] * data_spacing[0], data_array.shape[1] * data_spacing[1], data_array.shape[2] * \
              data_spacing[2]

    print("data_array.shape: ", data_array.shape)
    print("data_spacing: ", data_spacing)
    targetX, targetY, targetZ = int(x / targetXSpacing), data_array.shape[1], data_array.shape[2]
    targetSpacing = [x / targetX, y / targetY, z / targetZ]
    targetData = resample(data_array, [targetX, targetY, targetZ], False)
    targetLabel = resample(label_array, [targetX, targetY, targetZ], True)

    # Save the data and label.
    utils.save_image(targetData, data_image, config.kits_config.image_folder, filename, spacing=targetSpacing)
    utils.save_image(targetLabel, label_image, config.kits_config.label_folder, filename, spacing=targetSpacing)


if __name__ == '__main__':
    '''
    将原始Ablation数据的尺寸和Spacing进行转换，方便适配不同的模型。
    nohup python -u pangteen/kits/kits_transform.py > main0.out 2>&1 &
    '''
    dataset_folder = config.kits_config.dataset_folder

    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()

    for filename in utils.next_file(dataset_folder, sort=True):
        if filename.endswith('.json'):
            continue
        p.apply_async(transform, args=(dataset_folder, filename))


    p.close()
    p.join()
    print("=========> Finish all transforms ! Cost {} seconds.".format(time() - start_time))
