import os.path
from collections import OrderedDict

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
import pandas as pd
from skimage.transform import resize

from pangteen import config, utils
from pangteen.config import my_image_folder

if __name__ == '__main__':
    '''
    将原始Ablation数据的尺寸和Spacing进行转换，方便适配不同的模型。 
    python -u pangteen/ablation/test_transform.py
    '''
    origin_folder = config.ablation_origin_split_folder
    target_folder = config.my_split_folder

    table_path = os.path.join(config.my_base_folder, "ablation_mapping.xlsx")
    table = pd.read_excel(table_path)
    utils.maybe_mkdir(target_folder)
    for filename in utils.next_file(origin_folder):
        image, array, _ = utils.read_image(origin_folder, filename)
        row = table.loc[table["origin_name"] == filename]
        target_filename = row["new_name"].values[0]
        print(target_filename)
        utils.save_image(array, image, target_folder, target_filename)

    print("Finish transform all test split data.")

