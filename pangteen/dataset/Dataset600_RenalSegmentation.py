import random
from cProfile import label

from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from pangteen import config, utils
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from pangteen.renal import setting


# Dataset 101: 原始 XRay。
def convert_ablation(image_folder_name: str, label_folder_name: str, nnunet_dataset_id: int = 600):
    task_name = "MyAblation"
    task_config = config.renal_config

    image_folder = join(task_config.origin_folder, image_folder_name)
    label_folder = join(task_config.origin_folder, label_folder_name)
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    image_str = join(out_base, "imagesTr")
    test_str = join(out_base, "imagesTs")
    label_str = join(out_base, "labelsTr")
    maybe_mkdir_p(image_str)
    maybe_mkdir_p(test_str)
    maybe_mkdir_p(label_str)

    valid_ids = setting.get_valid_ids(70)


    image_filenames = subfiles(image_folder, join=False)
    test_ids = []
    train_cnt = 0
    # 顺序打乱
    random.shuffle(image_filenames)

    # 排除测试集的训练集。
    for image_filename in image_filenames:
        case_id = setting.get_case_id(image_filename)
        if case_id not in valid_ids:
            continue
        if len(test_ids) < setting.test_set_num:
            test_ids.append(image_filename)
            shutil.copy(join(image_folder, image_filename), join(test_str, "case_{:03d}_0000.nii.gz".format(case_id)))
            print("Finish test set of {}".format(image_filename))
        else:
            train_cnt += 1
            label_filename = image_filename.replace("image", "kidney_with_artery")
            shutil.copy(join(label_folder, label_filename), join(label_str, "case_{:03d}.nii.gz".format(case_id)))
            shutil.copy(join(image_folder, image_filename),join(image_str, "case_{:03d}_0000.nii.gz".format(case_id)))
            print("Finish train set of {}".format(image_filename))

    print("Train data : {}".format(train_cnt))

    generate_dataset_json(out_base,
                          channel_names={
                              0: "CT"
                          },
                          labels={
                              "background": 0,
                              "artery": 1,
                              "right kidney": 2,
                              "left kidney": 3
                          },
                          num_training_cases=train_cnt, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='1.0.0',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="RenalSegmentation")


if __name__ == '__main__':
    '''
    python -u pangteen/dataset/Dataset600_RenalSegmentation.py image kidney_with_artery -d 666
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help="The image folder of this dataset")
    parser.add_argument('label_folder', type=str, help="The label folder of this dataset")
    parser.add_argument('-d', required=False, type=int, default=200, help='Dataset ID, default: 200')
    args = parser.parse_args()
    convert_ablation(args.input_folder, args.label_folder, args.d)

