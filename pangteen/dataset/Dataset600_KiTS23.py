import random
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from pangteen import config, utils

def conver_test_set(kits_base_dir, case_id, test_str):
    shutil.copy(join(kits_base_dir, case_id, 'imaging.nii.gz'), join(test_str, f'{case_id}_0000.nii.gz'))
    print("Finish test set of {}".format(case_id))

def conver_train_set(kits_base_dir, case_id, image_str, label_str):
    shutil.copy(join(kits_base_dir, case_id, 'imaging.nii.gz'), join(image_str, f'{case_id}_0000.nii.gz'))
    shutil.copy(join(kits_base_dir, case_id, 'segmentation.nii.gz'), join(label_str, f'{case_id}.nii.gz'))
    print("Finish train set of {}".format(case_id))

# Dataset 801: KiTS2023，期待有官方测试集。
# Dataset 888：KiTS2023，没有官方测试集。
def convert_kits2023(dataset_folder_name: str, nnunet_dataset_id: int = 800):
    task_config = config.kits_config
    task_name = "KiTS2023"

    kits_base_dir = join(task_config.base_folder, dataset_folder_name)
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
    test_cnt = 100
    train_count = 0

    test_set_ids = []
    train_set_ids = []
    for i in cases:
        image_id = i
        label_id = i
        if test_cnt > 0:
            test_cnt -= 1
            test_set_ids.append(image_id)
        else:
            train_set_ids.append(image_id)
            train_count += 1

    # 多进程
    p = Pool(config.max_cpu_cnt)
    for i in test_set_ids:
        p.apply_async(conver_test_set, args=(kits_base_dir, i, test_str))
    for i in train_set_ids:
        p.apply_async(conver_train_set, args=(kits_base_dir, i, image_str, label_str))

    p.close()
    p.join()
    print("Train data : {}".format(train_count))

    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "kidney": (1, 2, 3),
                              "masses": (2, 3),
                              "tumor": 2
                          },
                          regions_class_order=(1, 3, 2),
                          num_training_cases=train_count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='0.1.3',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="KiTS2023")


if __name__ == '__main__':
    '''
    python -u pangteen/dataset/Dataset600_KiTS23.py image label -d 666
    nohup python -u pangteen/dataset/Dataset600_KiTS23.py dataset -d 666 > main3.out 2>&1 &
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help="The image folder of this dataset")
    parser.add_argument('-d', required=False, type=int, default=666, help='Dataset ID, default: 200')
    args = parser.parse_args()
    convert_kits2023(args.input_folder, args.d)


