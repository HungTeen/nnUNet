from os.path import join

from pangteen import config, utils
from pangteen.renal import setting


def combine_label():
    task_config = config.renal_config
    for filename in utils.next_file(task_config.image_folder):
        case_id = setting.get_case_id(filename)
        if case_id in setting.black_list:
            continue
        artery_image, artery_array, _ = utils.read_image(join(task_config.origin_folder, 'artery'), "renal_artery_mask_{:03d}.nii.gz".format(case_id))
        kidney_image, kidney_array, _ = utils.read_image(join(task_config.origin_folder, 'kidney'), "kidney_mask_{:03d}.nii.gz".format(case_id))
        kidney_array[artery_array == 1] = 1
        utils.save_image(kidney_array, kidney_image, join(task_config.origin_folder, 'kidney_with_artery'), "kidney_with_artery_{:03d}.nii.gz".format(case_id))
        print("Finish combine label of {}".format(filename))


if __name__ == '__main__':
    combine_label()