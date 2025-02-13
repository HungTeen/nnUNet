import os.path
from collections import OrderedDict

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
import pandas as pd
from skimage.transform import resize

from pangteen import config, utils

def test_transform(origin_folder, target_folder, table_folder, table_name="ablation_mapping.xlsx"):
    table = pd.read_excel(os.path.join(table_folder, table_name))
    utils.maybe_mkdir(target_folder)
    for filename in utils.next_file(origin_folder):
        image, array, _ = utils.read_image(origin_folder, filename)
        row = table.loc[table["origin_name"] == filename]
        target_filename = row["new_name"].values[0]
        print(target_filename)
        utils.save_image(array, image, target_folder, target_filename)

    print("Finish transform all test split data.")


def label_transform():
    folder = config.liver_ablation_config.label_folder
    for filename in utils.next_file(folder):
        case_id = int(filename.split('_')[1].split('.')[0])
        if case_id < 84:
            continue
        image, array, _ = utils.read_image(folder, filename)
        array[array == 1] = 2
        utils.save_image(array, image, folder, filename)

if __name__ == '__main__':
    '''
    对测试集的数据进行重命名。
    python -u pangteen/ablation/test_transform.py
    '''
    test_transform(config.tumor_ablation_config.test_split_folder,
                   config.tumor_ablation_config.renamed_test_split_folder,
                   config.tumor_ablation_config.base_folder)

