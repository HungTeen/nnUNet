import os
from fileinput import filename
from os.path import join
from typing import Optional

from pangteen.renal import setting

# 参数设置。
max_cpu_cnt = 6
drop_out_rate = None # 控制 nnUNet 各个卷积层是否使用 dropout。

# 文件夹相关。
def get_base_folder():
    base_folder = join('/data', 'ht')  # .98 服务器。
    if not os.path.exists(base_folder):
        base_folder = join('/mnt', 'data1', 'ht')  # .99 服务器。
    return base_folder
my_folder = get_base_folder()
predict_folder = join(my_folder, 'predict')  # 存放模型预测结果。
predict_train_folder = join(my_folder, 'predict_train')  # 存放模型预测结果。
val_id_set = [4, 6, 38, 42, 59, 62, 65, 68, 70, 78, 79]  # 测试集的 id。
test_id_set = [2, 3, 10, 12, 13, 15, 20, 28, 32, 34, 36, 37, 44, 47, 48, 57, 61, 71, 73, 76]  # 测试集的 id。

# 任务设置相关
config_list = []

class BaseConfig:

    def __init__(self, task_name: str, task_patch_size: tuple, label_map: dict, folder_name : Optional[str] = None):
        self.task_name = task_name
        self.patch_size = task_patch_size
        self.label_map = label_map
        self.folder_name = folder_name if folder_name else task_name
        self.base_folder = join(my_folder, self.folder_name)  # 项目数据的根路径。
        config_list.append(self)

    def get_label_path(self, image_filename):
        raise NotImplementedError('gt_folder() is not implemented.')

class AblationConfig(BaseConfig):
    """
    联影原始数据集任务。
    """

    def __init__(self):
        super().__init__(task_name='Ablation', task_patch_size=(64, 128, 128), label_map={
            '0': 'background',
            '1': 'area',
        })
        self.origin_folder = join(self.base_folder, 'origin')  # 存放最原始的图像，未经任何预处理。
        self.origin_image_folder = join(self.origin_folder, 'data')
        self.origin_label_folder = join(self.origin_folder, 'label')
        self.origin_contrast_folder = join(self.origin_folder, 'contrast')
        self.origin_split_folder = join(self.origin_folder, 'test_split')
        self.image_folder = join(self.base_folder, 'image')
        self.label_folder = join(self.base_folder, 'label')

    def get_label_path(self, image_filename):
        return join(self.label_folder, image_filename)

class MyAblationConfig(BaseConfig):
    """
    经过预处理的数据集任务。
    """

    def __init__(self):
        super().__init__(task_name='MyAblation', task_patch_size=(64, 128, 128), label_map={
            '0': 'background',
            '1': 'area',
        })
        self.my_image_folder = join(self.base_folder, 'image')
        self.my_label_folder = join(self.base_folder, 'label')
        self.my_split_folder = join(self.base_folder, 'test_split')

    def get_label_path(self, image_filename):
        return join(self.my_label_folder, image_filename)


class MyTumorConfig(BaseConfig):

    def __init__(self):
        super().__init__(task_name='MyTumor', task_patch_size=(128, 128, 128), label_map={
            '0': 'background',
            '1': 'tumor',
        }, folder_name='Ablation')
        self.origin_folder = join(self.base_folder, 'origin')  # 存放最原始的图像，未经任何预处理。
        self.image_folder = join(self.origin_folder, 'tumor')
        self.label_folder = join(self.origin_folder, 'tumor_label')


    def get_label_path(self, image_filename):
        return join(self.label_folder, image_filename)


class LiTSConfig(BaseConfig):

    def __init__(self):
        super().__init__(task_name='MyLiTS', task_patch_size=(128, 128, 128), label_map={
            '0': 'background',
            '1': 'liver',
            '2': 'tumor',
        }, folder_name='TumorAblation')
        self.image_folder = join(self.base_folder, 'liver_image')
        self.label_folder = join(self.base_folder, 'liver_label')
        self.cut_image_folder = join(self.base_folder, 'cut_liver_image')
        self.cut_label_folder = join(self.base_folder, 'cut_liver_label')


    def get_label_path(self, image_filename):
        return join(self.cut_label_folder, image_filename)


class TumorAblationConfig(BaseConfig):
    """
    肝脏+肿瘤+消融区域分割任务。
    """

    def __init__(self):
        super().__init__(task_name='TumorAblation', task_patch_size=(128, 128, 128), label_map={
            '0': 'background',
            '1': 'liver',
            '2': 'tumor',
            '3': 'ablation area',
        })
        self.rescale_image_folder = join(self.base_folder, 'rescale_image')
        self.rescale_label_folder = join(self.base_folder, 'rescale_label')
        self.predict_folder = join(self.base_folder, 'predict_tumor_label')
        self.compose_label_folder = join(self.base_folder, 'compose_label')  # 拼接而成的标签。
        self.image_folder = join(self.base_folder, 'renamed_image')
        self.label_folder = join(self.base_folder, 'renamed_label')
        self.test_split_folder = join(self.base_folder, 'test_split')
        self.renamed_test_split_folder = join(self.base_folder, 'renamed_test_split')


    def get_label_path(self, image_filename):
        return join(self.label_folder, image_filename)


class LiverAblationConfig(BaseConfig):
    """
    消融区域分割任务。
    """

    def __init__(self):
        super().__init__(task_name='LiverAblation', task_patch_size=(128, 128, 128), label_map={
            '0': 'background',
            '1': 'ablation area',
            '2': 'history ablation area',
        })
        self.history_image_folder = join(self.base_folder, 'history_image')
        self.history_image_for_predict_folder = join(self.base_folder, 'history_image_for_predict')
        self.history_label_folder = join(self.base_folder, 'history_label')
        self.compose_label_folder = join(self.base_folder, 'compose_label')  # 拼接而成的标签。
        self.image_folder = join(self.base_folder, 'renamed_image')
        self.label_folder = join(self.base_folder, 'renamed_label')
        self.test_split_folder = join(self.base_folder, 'test_split')
        self.renamed_test_split_folder = join(self.base_folder, 'renamed_test_split')


    def get_label_path(self, image_filename):
        return join(self.label_folder, image_filename)

class RenalConfig(BaseConfig):
    """
    肾脏+动脉分割任务。
    """
    def __init__(self):
        super().__init__(task_name='RenalSegmentation', task_patch_size=(128, 128, 128), label_map={
            '0': 'background',
            '1': 'artery',
            '2': 'right kidney',
            '3': 'left kidney',
        })
        self.origin_folder = join(self.base_folder, 'origin')  # 存放最原始的图像，未经任何预处理。
        self.image_folder = join(self.origin_folder, 'image')
        self.label_folder = join(self.origin_folder, 'kidney_with_artery')

    def get_label_path(self, image_filename):
        filename = image_filename.replace('case', 'kidney_with_artery')
        return join(self.label_folder, filename)

class KiTSConfig(BaseConfig):
    """
    Kidney Tumor Segmentation Challenge 2023。
    """

    def __init__(self):
        super().__init__(task_name='KiTS', task_patch_size=(128, 128, 128), label_map={
            '0': 'background',
            '1': 'kidney',
            '2': 'tumor',
            '3': 'masses',
        })
        self.dataset_folder = join(self.base_folder, 'dataset')  # 存放原始数据。

    def get_label_path(self, image_filename):
        case_name = image_filename.split('.')[0]
        return join(self.dataset_folder, case_name, 'segmentation.nii.gz')

# 任务设置。
ablation_config = AblationConfig()
my_ablation_config = MyAblationConfig()
my_tumor_config = MyTumorConfig()
lits_config = LiTSConfig()
tumor_ablation_config = TumorAblationConfig()
liver_ablation_config = LiverAblationConfig()
renal_config = RenalConfig()
kits_config = KiTSConfig()
main_config: BaseConfig = lits_config
patch_size = main_config.patch_size
label_map = main_config.label_map
