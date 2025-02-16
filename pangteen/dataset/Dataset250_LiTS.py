import random

from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from pangteen import config, utils
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# Dataset 256: LiTS
def convert_ablation(image_folder_name: str, label_folder_name: str, nnunet_dataset_id: int = 250):
    task_config = config.lits_config
    task_name = task_config.task_name

    image_folder = join(task_config.base_folder, image_folder_name)
    label_folder = join(task_config.base_folder, label_folder_name)
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    image_str = join(out_base, "imagesTr")
    test_str = join(out_base, "imagesTs")
    label_str = join(out_base, "labelsTr")
    maybe_mkdir_p(image_str)
    maybe_mkdir_p(test_str)
    maybe_mkdir_p(label_str)

    image_ids = subfiles(image_folder, join=False)
    random.shuffle(image_ids)
    test_count = 30
    train_count = 0

    for i in image_ids:
        image_id = i
        label_id = i
        if test_count > 0:
            test_count -= 1
            shutil.copy(join(image_folder, image_id), join(test_str, utils.append_name(image_id, "_0000")))
            print("Finish test set of {}".format(image_id))
        else:
            shutil.copy(join(label_folder, label_id), join(label_str, label_id))
            shutil.copy(join(image_folder, image_id), join(image_str, utils.append_name(image_id, "_0000")))
            print("Finish train set of {}".format(image_id))
            train_count += 1

    print("Train data : {}".format(train_count))

    generate_dataset_json(out_base,
                          channel_names={
                              0: "CT"
                          },
                          labels={
                              "background": 0,
                              "liver": 1,
                              "tumor": 2
                          },
                          num_training_cases=train_count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='1.0.0',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="LiTS")


if __name__ == '__main__':
    '''
    python -u pangteen/dataset/Dataset250_LiTS.py cut_liver_image cut_liver_label -d 256
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help="The image folder of this dataset")
    parser.add_argument('label_folder', type=str, help="The label folder of this dataset")
    parser.add_argument('-d', required=False, type=int, default=200, help='Dataset ID, default: 200')
    args = parser.parse_args()
    convert_ablation(args.input_folder, args.label_folder, args.d)

