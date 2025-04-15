import os.path
from collections import OrderedDict
from multiprocessing import Pool
from time import time

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates, rotate
import pandas as pd
from skimage.transform import resize

from pangteen import config, utils
from pangteen.util.resample import resample

def transform(origin_image_folder, origin_label_folder, filename, target_filename):
    print("Processing: ", target_filename)

    filenames = os.listdir(origin_image_folder)
    if os.path.exists(os.path.join(origin_image_folder, target_filename)):
        print("Already exists: ", target_filename)
        return


    data_image, data_array, data_spacing = utils.read_image(origin_image_folder, filename)
    label_image, label_array, label_spacing = utils.read_image(origin_label_folder, filename)
    assert data_spacing[0] == label_spacing[0] and data_spacing[1] == label_spacing[1] and data_spacing[2] == \
           label_spacing[2]
    assert data_array.shape == label_array.shape
    # Resize the data and label.
    print("Transform: ", filename + " to " + target_filename)

    # # 随机生成 0 - 5 度的旋转角度
    # angle = np.random.uniform(0, 5)
    #
    # # 对数据和标签进行旋转
    # data_array = rotate(data_array, angle, axes=(1, 2), reshape=False, order=3)
    # label_array = rotate(label_array, angle, axes=(1, 2), reshape=False, order=0)

    # Save the data and label.
    utils.save_image(data_array, data_image, origin_image_folder, target_filename)
    utils.save_image(label_array, label_image, origin_label_folder, target_filename)


if __name__ == '__main__':
    '''
    将原始Ablation数据的尺寸和Spacing进行转换，方便适配不同的模型。
    python -u pangteen/ablation/ablation_transform.py
    nohup python -u pangteen/ablation/ablation_transform.py > main3.out 2>&1 &
    '''
    origin_image_folder = config.lits_config.cut_image_folder
    origin_label_folder = config.lits_config.cut_label_folder

    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()

    start_id = 131
    gen_num = 29

    filenames = [i for i in range(start_id)]
    # 打乱 filenames 的顺序
    np.random.shuffle(filenames)
    for id in range(gen_num):
        case_id = id + start_id
        filename = 'liver_{:03d}.nii.gz'.format(filenames[id])
        target_filename = 'liver_{:03d}.nii.gz'.format(case_id)

        p.apply_async(transform, args=(origin_image_folder, origin_label_folder, filename, target_filename))


    p.close()
    p.join()
    print("=========> Finish all transforms ! Cost {} seconds.".format(time() - start_time))
