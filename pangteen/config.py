from os.path import join

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
origin_label_folder = join(ablation_origin_folder, 'label')
ablation_origin_contrast_folder = join(ablation_origin_folder, 'contrast')
ablation_origin_split_folder = join(ablation_origin_folder, 'test_split')
ablation_predict_folder = join(ablation_base_folder, 'predict')  # 存放模型预测结果。