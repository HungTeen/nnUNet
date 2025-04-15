import os
import shutil
from multiprocessing.pool import ThreadPool

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import SimpleITK as sitk
from pangteen import utils, config


def port(path, target_path, trainer_name):
    fold_path = os.path.join(path, trainer_name, 'fold_0')
    best_params_name = 'checkpoint_best.pth'
    final_params_name = 'checkpoint_final.pth'
    progress_name = 'progress.png'
    print("Start Porting trainer: ", trainer_name)

    # 构建源文件路径
    best_params_src = os.path.join(fold_path, best_params_name)
    final_params_src = os.path.join(fold_path, final_params_name)
    progress_src = os.path.join(fold_path, progress_name)

    target_trainer_name = trainer_name.split('_')[0]
    # 构建目标文件路径
    target_dst = os.path.join(target_path, target_trainer_name)

    utils.maybe_mkdir(target_dst)

    try:
        # 复制最佳参数文件
        shutil.copy2(best_params_src, target_dst)
        print(f"Copied {best_params_name} to {target_dst}")
        # 复制最终参数文件
        shutil.copy2(final_params_src, target_dst)
        print(f"Copied {final_params_name} to {target_dst}")
        # 复制进度文件
        shutil.copy2(progress_src, target_dst)
        print(f"Copied {progress_name} to {target_dst}")
    except FileNotFoundError:
        print("One or both of the parameter files were not found.")
    except Exception as e:
        print(f"An error occurred during copying: {e}")


def port_models(path, target_path, name_mapping):
    # 多线程
    pool = ThreadPool(processes=config.max_cpu_cnt)
    # 将图像裁剪为同样的大小
    for filename in os.listdir(path):
        trainer_name = filename.split('_')[0]
        model_name = trainer_name.split('Trainer')[0]
        if model_name in name_mapping.keys():
            replace_name = name_mapping.get(model_name)
            filename = filename.replace(model_name, replace_name)

        pool.apply_async(port, (path, target_path, filename))
        break

    pool.close()
    pool.join()
    print("All models have been ported.")

if __name__ == '__main__':
    path = os.path.abspath('D:\Dataset256_MyLiTS')
    target_path = os.path.abspath('D:\models')
    name_mapping = {
        'KD': 'SKIMUNet',
        'MambaAll': 'MambaFusionV1',
        'HT': 'nnUNet',
        'DSUKan': 'SBCNet',
        'XTUKan': 'TDNet',
        'ResMambaAll': 'RepUXNet',
        'GSCUKan': 'MLPUKan',
        'EPAUKan': 'ShiftedMLPUKan',
        'SFEPAUKan': 'NoShiftedUKan',
    }
    port_models(path, target_path, name_mapping)
    pass