from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from pangteen import config, utils
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# Dataset 251: MyTumor
# Dataset 256: LiTS
def convert_ablation(image_folder_name: str, label_folder_name: str, nnunet_dataset_id: int = 250):
    task_config = config.lits_config
    task_name = task_config.task_name

    image_folder = join(task_config.base_folder, image_folder_name)
    label_folder = join(task_config.base_folder, label_folder_name)
    test_folder = config.tumor_ablation_config.rescale_image_folder  # 需要预测的数据集。
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    image_str = join(out_base, "imagesTr")
    test_str = join(out_base, "imagesTs")
    label_str = join(out_base, "labelsTr")
    maybe_mkdir_p(image_str)
    maybe_mkdir_p(test_str)
    maybe_mkdir_p(label_str)

    image_ids = subfiles(image_folder, join=False)
    test_ids = subfiles(test_folder, join=False)

    # 使用预先划分好的测试集。
    for i in test_ids:
        image_id = i
        shutil.copy(join(test_folder, image_id), join(test_str, utils.append_name(image_id, "_0000")))
        print("Finish test set of {}".format(image_id))

    print("Test data : {}".format(len(test_ids)))

    # 排除测试集的训练集。
    for i in image_ids:
        image_id = i
        label_id = i
        shutil.copy(join(label_folder, label_id), join(label_str, label_id))
        shutil.copy(join(image_folder, image_id), join(image_str, utils.append_name(image_id, "_0000")))
        print("Finish train set of {}".format(image_id))

    train_len = len(image_ids)
    print("Train data : {}".format(train_len))

    generate_dataset_json(out_base,
                          channel_names={
                              0: "CT"
                          },
                          labels={
                              "background": 0,
                              "liver": 1,
                              "tumor": 2
                          },
                          num_training_cases=train_len, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='1.0.0',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="Ablation")


if __name__ == '__main__':
    '''
    python -u pangteen/dataset/Dataset250_MyTumor.py cut_liver_image cut_liver_label -d 256
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help="The image folder of this dataset")
    parser.add_argument('label_folder', type=str, help="The label folder of this dataset")
    parser.add_argument('-d', required=False, type=int, default=200, help='Dataset ID, default: 200')
    args = parser.parse_args()
    convert_ablation(args.input_folder, args.label_folder, args.d)

