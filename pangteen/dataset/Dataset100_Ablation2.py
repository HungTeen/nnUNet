from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from pangteen import config, utils
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

def convert_ablation_deprecated(nnunet_dataset_id: int = 220):
    """
    双通道（CT+肝脏mask）数据集转换为nnUNet数据集。
    """
    task_name = "Ablation"

    image_folder = config.tumor_ablation_config.rescale_image_folder
    image2_folder = config.tumor_ablation_config.predict_folder
    label_folder = config.tumor_ablation_config.rescale_label_folder
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    image_str = join(out_base, "imagesTr")
    test_str = join(out_base, "imagesTs")
    label_str = join(out_base, "labelsTr")
    maybe_mkdir_p(image_str)
    maybe_mkdir_p(test_str)
    maybe_mkdir_p(label_str)

    image_ids = subfiles(image_folder, join=False)
    test_ids = subfiles(config.tumor_ablation_config.test_split_folder, join=False)

    # 使用预先划分好的测试集。
    for i in test_ids:
        image_id = i
        shutil.copy(join(image_folder, image_id), join(test_str, utils.append_name(image_id, "_0000")))
        shutil.copy(join(image2_folder, image_id), join(test_str, utils.append_name(image_id, "_0001")))
        print("Finish test set of {}".format(image_id))

    print("Test data : {}".format(len(test_ids)))

    # 排除测试集的训练集。
    for i in image_ids:
        if i in test_ids:
            continue
        image_id = i
        label_id = i
        shutil.copy(join(label_folder, label_id), join(label_str, label_id))
        shutil.copy(join(image_folder, image_id), join(image_str, utils.append_name(image_id, "_0000")))
        shutil.copy(join(image2_folder, image_id), join(image_str, utils.append_name(image_id, "_0001")))
        print("Finish train set of {}".format(image_id))

    train_len = len(image_ids) - len(test_ids)
    print("Train data : {}".format(train_len))

    generate_dataset_json(out_base,
                          channel_names={
                              0: "CT",
                              1: "LiverMask"
                          },
                          labels={
                              "background": 0,
                              "ablation": 1
                          },
                          num_training_cases=train_len, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='1.0.0',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="Ablation")

def convert_ablation(nnunet_dataset_id: int = 100):
    """
    """
    task_name = "Ablation"

    image_folder = config.ablation_config.image_folder
    label_folder = config.ablation_config.label_folder
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    image_str = join(out_base, "imagesTr")
    test_str = join(out_base, "imagesTs")
    label_str = join(out_base, "labelsTr")
    maybe_mkdir_p(image_str)
    maybe_mkdir_p(test_str)
    maybe_mkdir_p(label_str)

    image_ids = subfiles(image_folder, join=False)
    test_ids = subfiles(config.tumor_ablation_config.test_split_folder, join=False)

    for i in image_ids:
        image_id = i
        label_id = i
        case_id = int(image_id.split('_')[1].split('.')[0])
        if case_id in config.test_id_set:
            shutil.copy(join(image_folder, image_id), join(test_str, utils.append_name(image_id, "_0000")))
            print("Finish test set of {}".format(image_id))
        else:
            shutil.copy(join(label_folder, label_id), join(label_str, label_id))
            shutil.copy(join(image_folder, image_id), join(image_str, utils.append_name(image_id, "_0000")))
            print("Finish train set of {}".format(image_id))

    train_len = len(image_ids) - len(test_ids)
    print("Train data : {}".format(train_len))

    generate_dataset_json(out_base,
                          channel_names={
                              0: "CT",
                          },
                          labels={
                              "background": 0,
                              "ablation": 1
                          },
                          num_training_cases=train_len, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='1.0.0',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="Ablation")


if __name__ == '__main__':
    """
    python -u pangteen/dataset/Dataset100_Ablation2.py -d 108
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', required=False, type=int, default=100, help='Dataset ID, default: 100')
    args = parser.parse_args()
    convert_ablation(args.d)

