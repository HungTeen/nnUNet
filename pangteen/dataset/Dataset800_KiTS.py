import random

from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from pangteen.kits import config
from pangteen import config as pang_config

# Dataset 801: KiTS2023，期待有官方测试集。
# Dataset 888：KiTS2023，没有官方测试集。
def convert_kits2023(kits_base_dir, nnunet_dataset_id: int = 800, no_official_testset: bool = False):
    task_name = "KiTS2023"

    folder_name = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, folder_name)
    image_str = join(out_base, "imagesTr")
    label_str = join(out_base, "labelsTr")
    test_str = join(out_base, "imagesTs")
    maybe_mkdir_p(image_str)
    maybe_mkdir_p(label_str)
    maybe_mkdir_p(test_str)

    cases = subdirs(kits_base_dir, prefix='case_', join=False)
    random.shuffle(cases)
    test_cnt = 0
    training_case_num = 0
    for tr in cases:
        # 如果存在文件
        if not isfile(join(kits_base_dir, tr, 'imaging.nii.gz')):
            continue
        if test_cnt < config.TEST_CASE_COUNT:
            test_cnt += 1
            shutil.copy(join(kits_base_dir, tr, 'imaging.nii.gz'), join(test_str, f'{tr}_0000.nii.gz'))
            print("Test set of {}".format(tr))
        else:
            training_case_num += 1
            shutil.copy(join(kits_base_dir, tr, 'imaging.nii.gz'), join(image_str, f'{tr}_0000.nii.gz'))
            shutil.copy(join(kits_base_dir, tr, 'segmentation.nii.gz'), join(label_str, f'{tr}.nii.gz'))
            print("Train set of {}".format(tr))

    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "kidney": (1, 2, 3),
                              "masses": (2, 3),
                              "tumor": 2
                          },
                          regions_class_order=(1, 3, 2),
                          num_training_cases=training_case_num, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='0.1.3',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="KiTS2023")


if __name__ == '__main__':
    '''
    python -u pangteen/dataset/Dataset800_KiTS.py dataset -d 801
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help="The dataset folder of KiTS2023")
    parser.add_argument('-d', required=False, type=int, default=800, help='Dataset ID, default: 800')
    parser.add_argument('--no_official_testset', action='store_true',
                        help='Do not include the official test set in the dataset')
    args = parser.parse_args()

    image_folder = join(pang_config.kits_config.base_folder, args.input_folder)
    convert_kits2023(image_folder, args.d, args.no_official_testset)


