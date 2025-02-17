import random

from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from pangteen import config, utils
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

def convert_ablation():
    task_name = "SHR"

    image_folder = join(config.my_folder, 'shr', 'images')
    label_folder = join(config.my_folder, 'shr', 'labels')
    foldername = "Dataset%03.0d_%s" % (999, task_name)

    out_base = join(nnUNet_raw, foldername)
    image_str = join(out_base, "imagesTr")
    test_str = join(out_base, "imagesTs")
    label_str = join(out_base, "labelsTr")
    maybe_mkdir_p(image_str)
    maybe_mkdir_p(test_str)
    maybe_mkdir_p(label_str)

    image_ids = subfiles(image_folder, join=False)

    train_count = 0

    # 排除测试集的训练集。
    for i in image_ids:
        image_id = i
        name = image_id.split('.')[0]
        label_id = name + '_watershed_mask.png'
        if random.random() < 0.15:
            shutil.copy(join(image_folder, image_id), join(test_str, utils.append_png_name(image_id, "_0000")))
            print("Finish test set of {}".format(image_id))
        else:
            shutil.copy(join(label_folder, label_id), join(label_str, image_id))
            shutil.copy(join(image_folder, image_id), join(image_str, utils.append_png_name(image_id, "_0000")))
            print("Finish train set of {}".format(image_id))
            train_count += 1

    print("Train data : {}".format(train_count))

    generate_dataset_json(out_base,
                          channel_names={0: 'R', 1: 'G', 2: 'B'},
                          labels={
                              "background": 0,
                              "label 1": 1,
                              "label 2": 2,
                              "label 3": 3,
                              "label 4": 4,
                              "label 5": 5,
                              "label 6": 6,
                              "label 7": 7,
                              "label 8": 8,
                              "label 9": 9,
                              "label 10": 10,
                          },
                          num_training_cases=train_count, file_ending='.png',
                          dataset_name=task_name, reference='none',
                          release='1.0.0',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="Ablation")


if __name__ == '__main__':
    """
    python -u pangteen.dataset.Dataset900_SHR
    """
    convert_ablation()

