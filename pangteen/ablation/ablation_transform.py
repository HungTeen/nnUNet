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
from pangteen.config import my_image_folder


def resample(data, target_shape, is_seg):
    data = np.array([data])
    return resample_data_or_seg(data, target_shape, is_seg)[0]


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


def transform(origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename, target_filename):
    print("Processing: ", filename, " -> ", target_filename)
    targetXYSize = 256
    targetZSpacing = 2.5
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

if __name__ == '__main__':
    '''
    将原始Ablation数据的尺寸和Spacing进行转换，方便适配不同的模型。
    python -u pangteen/ablation/ablation_transform.py
    nohup python -u pangteen/ablation/ablation_transform.py > main3.out 2>&1 &
    '''
    origin_image_folder = config.ablation_origin_image_folder
    origin_label_folder = config.ablation_origin_label_folder
    target_image_folder = config.my_image_folder
    target_label_folder = config.my_label_folder

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

        p.apply_async(transform, args=(origin_image_folder, origin_label_folder, target_image_folder, target_label_folder, filename, target_filename))

    pd.DataFrame(mapping_table, index=rows, columns=['new_name']).to_excel('ablation_mapping.xlsx')
    p.close()
    p.join()
    print("=========> Finish all transforms ! Cost {} seconds.".format(time() - start_time))