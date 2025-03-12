import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import SimpleITK as sitk
from pangteen import utils

def visualize_and_save(ct_path, label_path, case_name, output_path, output_name, slice_index=None):
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
    overlay[label_slice == 1] = red_color
    overlay[label_slice == 2] = green_color

    # 将标签叠加在 CT 图像上
    alpha = 0.6  # 透明度
    combined = (1 - alpha) * ct_rgb + alpha * overlay

    # 显示叠加后的图像
    plt.imshow(combined)
    plt.axis('off')

    # 保存图像
    plt.savefig(os.path.join(output_path, output_name), bbox_inches='tight', pad_inches=0)
    plt.close()


def cut_cmp_image_and_label(cmp_id_map: dict, size=(400, 512)):
    ct_path = os.path.abspath('E:\Study\LiTS\cut_liver_image')
    gt_path = os.path.abspath('E:\Study\LiTS\cut_liver_label')
    cmp_path = os.path.abspath('E:\Study\LiTS\predict')
    output_path = os.path.abspath('E:\Study\研究生\截图\cmp')

    for case_id, case_slice in cmp_id_map.items():
        case_filename = f'liver_{case_id:03d}.nii.gz'
        ct_name = f'ct_{case_id:03d}.png'
        gt_name = f'gt_{case_id:03d}.png'
        visualize_and_save(ct_path, None, case_filename, output_path, ct_name, slice_index=case_slice)
        visualize_and_save(ct_path, gt_path, case_filename, output_path, gt_name, slice_index=case_slice)
        for cmp_folder_name in os.listdir(cmp_path):
            cmp_predict_path = os.path.join(cmp_path, cmp_folder_name, 'fold_[0]')
            model_name = cmp_folder_name.split('_')[0].replace('Trainer', '')
            output_name = f'{model_name}_{case_id:03d}.png'
            visualize_and_save(ct_path, cmp_predict_path, case_filename, output_path, output_name, slice_index=case_slice)
            # break


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
    cut_cmp_image_and_label({
        22: 200,
        36: 36,
        38: 96,
        83: 251
    })
    # 使用示例
    # ct_path = 'E:\Study\LiTS\cut_liver_image\liver_001.nii.gz'
    # label_path = 'E:\Study\LiTS\cut_liver_label\liver_001.nii.gz'
    # output_path = 'E:\Study\研究生\截图\cmp\output_image.png'
    #
    # visualize_and_save(ct_path, label_path, output_path)
    pass