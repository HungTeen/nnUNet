from fileinput import filename
from os.path import join

from pangteen.renal import setting

# 参数设置。
max_cpu_cnt = 6
drop_out_rate = None # 控制 nnUNet 各个卷积层是否使用 dropout。

# 文件夹相关。
my_folder = join('/mnt', 'data1', 'ht')
predict_folder = join(my_folder, 'predict')  # 存放模型预测结果。

# 任务设置相关
config_list = []

class BaseConfig:

    def __init__(self, task_name: str, task_patch_size: tuple, label_map: dict):
        self.task_name = task_name
        self.patch_size = task_patch_size
        self.label_map = label_map
        self.base_folder = join(my_folder, task_name)  # 项目数据的根路径。
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

    def get_label_path(self, image_filename):
        return join(self.origin_label_folder, image_filename)

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

class RenalConfig(BaseConfig):
    """
    肾脏分割任务。
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
renal_config = RenalConfig()
kits_config = KiTSConfig()
main_config: BaseConfig = renal_config
patch_size = main_config.patch_size
label_map = main_config.label_map
