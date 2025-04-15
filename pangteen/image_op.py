import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import SimpleITK as sitk
from pangteen import utils

def visualize_and_save(ct_path, label_path, case_name, output_path, output_name, slice_index=None, revert_y=False, revert_x=False, alpha=0.6, mode='center', radius=128):
    # 读取 CT 图像和标签图像
    ct_image, ct_array, _ = utils.read_image(ct_path, case_name, transpose=False)
    if label_path is not None:
        label_image, label_array, _ = utils.read_image(label_path, case_name, transpose=False)
    else:
        label_array = np.zeros(ct_array.shape, dtype=ct_array.dtype)

    # 选择一个切片进行可视化（这里选择中间切片）
    if slice_index is None:
        slice_index = ct_array.shape[0] // 2
    ct_slice = ct_array[slice_index, :, :]
    label_slice = label_array[slice_index, :, :]

    corner_y = 0
    corner_x = 0
    if mode == 'lefttop':
        corner_y = ct_slice.shape[0] - radius * 2
    elif mode == 'center':
        # 计算中心点坐标
        center_y, center_x = np.array(ct_slice.shape) // 2
        corner_y = max(0, center_y - radius)
        corner_x = max(0, center_x - radius)
    elif mode == 'left':
        center_y, center_x = np.array(ct_slice.shape) // 2
        corner_y = max(0, center_y - radius)
    elif mode == 'top':
        center_y, center_x = np.array(ct_slice.shape) // 2
        corner_x = max(0, center_x - radius)
        corner_y = ct_slice.shape[0] - radius * 2

    ct_slice = ct_slice[corner_y:corner_y + radius * 2, corner_x:corner_x + radius * 2]
    label_slice = label_slice[corner_y:corner_y + radius * 2, corner_x:corner_x + radius * 2]

    # 归一化 CT 切片到 [0, 1] 范围
    ct_min = np.min(ct_slice)
    ct_max = np.max(ct_slice)
    ct_slice = (ct_slice - ct_min) / (ct_max - ct_min)

    # 创建一个与 CT 切片相同大小的 RGB 图像
    ct_rgb = np.stack((ct_slice, ct_slice, ct_slice), axis=-1)

    # 将标签 1 显示为红色，标签 2 显示为绿色
    red_color = [1, 0, 0]
    green_color = [0, 1, 0]
    overlay = np.zeros_like(ct_rgb, dtype=np.float32)
    # overlay[label_slice == 1] = red_color
    overlay[label_slice == 2] = red_color

    # 将标签叠加在 CT 图像上
    combined = (1 - alpha) * ct_rgb + alpha * overlay

    # 显示叠加后的图像
    if revert_y:
        combined = np.flip(combined, axis=0)
    if revert_x:
        combined = np.flip(combined, axis=1)
    plt.imshow(combined)
    plt.axis('off')

    # 保存图像
    plt.savefig(os.path.join(output_path, output_name), bbox_inches='tight', pad_inches=0)
    plt.close()


def cut_cmp_image_and_label(cmp_id_map: dict, revert_y_list=None, revert_x_list=None, alpha=0.6, modes=None, radius_list=None):
    ct_path = os.path.abspath('E:\Study\LiTS\cut_liver_image')
    gt_path = os.path.abspath('E:\Study\LiTS\cut_liver_label')
    cmp_path = os.path.abspath('E:\Study\LiTS\predict\cmp')
    output_path = os.path.abspath('E:\Study\研究生\截图\cmp')

    if modes is None:
        modes = ['center'] * len(cmp_id_map)

    if radius_list is None:
        radius_list = [128] * len(cmp_id_map)

    for iid, (case_id, case_slice) in enumerate(cmp_id_map.items()):
        case_filename = f'liver_{case_id:03d}.nii.gz'
        ct_name = f'ct_{case_id:03d}.png'
        gt_name = f'gt_{case_id:03d}.png'
        revert_y = True if revert_y_list is not None and case_id in revert_y_list else False
        revert_x = True if revert_x_list is not None and case_id in revert_x_list else False
        mode = modes[iid]
        radius = radius_list[iid]
        visualize_and_save(ct_path, None, case_filename, output_path, ct_name, slice_index=case_slice, revert_y=revert_y, revert_x=revert_x, alpha=alpha, mode=mode, radius=radius)
        visualize_and_save(ct_path, gt_path, case_filename, output_path, gt_name, slice_index=case_slice, revert_y=revert_y, revert_x=revert_x, alpha=alpha, mode=mode, radius=radius)
        for cmp_folder_name in os.listdir(cmp_path):
            cmp_predict_path = os.path.join(cmp_path, cmp_folder_name, 'fold_[0]')
            model_name = cmp_folder_name.split('_')[0].replace('Trainer', '')
            output_name = f'{model_name}_{case_id:03d}.png'
            visualize_and_save(ct_path, cmp_predict_path, case_filename, output_path, output_name, slice_index=case_slice, revert_y=revert_y, revert_x=revert_x, alpha=alpha, mode=mode, radius=radius)
            # break
        print(f"Cut case {case_id} done.")

def cut_ablation_image_and_label(cmp_id_map: dict, revert_y_list=None, revert_x_list=None, alpha=0.6, modes=None, radius_list=None):
    ct_path = os.path.abspath('E:\Study\LiTS\cut_liver_image')
    gt_path = os.path.abspath('E:\Study\LiTS\cut_liver_label')
    cmp_path = os.path.abspath('E:\Study\LiTS\predict\\ablation')
    output_path = os.path.abspath('E:\Study\研究生\截图\\ablation')

    if modes is None:
        modes = ['center'] * len(cmp_id_map)

    if radius_list is None:
        radius_list = [128] * len(cmp_id_map)

    for iid, (case_id, case_slice) in enumerate(cmp_id_map.items()):
        case_filename = f'liver_{case_id:03d}.nii.gz'
        ct_name = f'ct_{case_id:03d}.png'
        gt_name = f'gt_{case_id:03d}.png'
        revert_y = True if revert_y_list is not None and case_id in revert_y_list else False
        revert_x = True if revert_x_list is not None and case_id in revert_x_list else False
        mode = modes[iid]
        radius = radius_list[iid]

        visualize_and_save(ct_path, None, case_filename, output_path, ct_name, slice_index=case_slice,
                           revert_y=revert_y, revert_x=revert_x, alpha=alpha, mode=mode, radius=radius)
        visualize_and_save(ct_path, gt_path, case_filename, output_path, gt_name, slice_index=case_slice,
                           revert_y=revert_y, revert_x=revert_x, alpha=alpha, mode=mode, radius=radius)
        for cmp_folder_name in os.listdir(cmp_path):
            cmp_predict_path = os.path.join(cmp_path, cmp_folder_name, 'fold_[0]')
            model_name = cmp_folder_name.split('_')[0].replace('Trainer', '')
            output_name = f'{model_name}_{case_id:03d}.png'
            visualize_and_save(ct_path, cmp_predict_path, case_filename, output_path, output_name,
                               slice_index=case_slice, revert_y=revert_y, revert_x=revert_x, alpha=alpha, mode=mode, radius=radius)
            # break

        print(f"Cut case {case_id} done.")

def cut_image(path, target_path, size=(400, 512)):
    """
    把图像居中裁剪为同样的大小。
    """
    path = os.path.abspath(path)
    # 遍历目录下所有文件
    image_sizes = []
    for filename in os.listdir(path):
        print("Find file: ", filename)
        # 读取图像大小
        try:
            with Image.open(os.path.join(path, filename)) as img:
                width, height = img.size
                image_sizes.append([width, height])
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    image_sizes = np.array(image_sizes)
    print(np.min(image_sizes[:, 0]), np.min(image_sizes[:, 1]))

    # 将图像裁剪为同样的大小
    for filename in os.listdir(path):
        print("Cut file: ", filename)
        try:
            with Image.open(os.path.join(path, filename)) as img:
                width, height = img.size
                left = (width - size[0]) // 2
                top = (height - size[1]) // 2
                right = left + size[0]
                bottom = top + size[1]
                img = img.crop((left, top, right, bottom))
                img.save(os.path.join(path, filename))
        except Exception as e:
            print(f"Error cutting {filename}: {e}")

if __name__ == '__main__':
    # cut_image('E:\Study\研究生\截图\cmp', 'E:\Study\研究生\截图\cut_cmp')
    alpha = 0.6
    # cut_cmp_image_and_label({
    #     22: 200,
    #     36: 36,
    #     38: 96,
    #     83: 251
    # }, revert_y_list=[22, 36, 38], alpha=alpha, modes=['center', 'left', 'center', 'left'], radius_list=[128, 128, 128, 96])
    cut_ablation_image_and_label({
        20: 212,
        75: 205,
        83: 251,
        113: 191
    }, revert_y_list=[20, 75], revert_x_list=[75], alpha=alpha, modes=['lefttop', 'top', 'left', 'center'], radius_list=[128, 160, 96, 160])
    pass