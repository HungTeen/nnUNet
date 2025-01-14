from os.path import join

max_cpu_cnt = 6
ablation_patch_size = (64, 128, 128)
drop_out_rate = 0.1

# 标注相关。
area_label = 1
label_map = {
    '0': 'background',
    '1': 'area',
}

# 文件夹相关。
my_folder = join('/mnt', 'data1', 'ht')
ablation_task_name = 'Ablation'
ablation_base_folder = join(my_folder, ablation_task_name)  # 项目数据的根路径。
ablation_origin_folder = join(ablation_base_folder, 'origin')  # 存放最原始的图像，未经任何预处理。
ablation_origin_image_folder = join(ablation_origin_folder, 'data')
ablation_origin_label_folder = join(ablation_origin_folder, 'label')
ablation_origin_contrast_folder = join(ablation_origin_folder, 'contrast')
ablation_origin_split_folder = join(ablation_origin_folder, 'test_split')

my_task_name = 'MyAblation'
my_base_folder = join(my_folder, my_task_name)  # 项目数据的根路径。
my_image_folder = join(my_base_folder, 'image')
my_label_folder = join(my_base_folder, 'label')
my_split_folder = join(my_base_folder, 'test_split')

kits_task_name = "KiTS"
kits_base_folder = join(my_folder, kits_task_name)  # 项目数据的根路径。
kits_dataset_folder = join(kits_base_folder, 'dataset')  # 存放原始数据。

predict_folder = join(my_folder, 'predict')  # 存放模型预测结果。