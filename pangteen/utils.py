import os.path
from typing import Optional, Tuple, Any, Union

import numpy as np
from SimpleITK import Image
from numpy import ndarray, dtype
import SimpleITK as silk

def next_file(folder, sort=True, return_path=False):
    """
    遍历文件夹下的所有文件。
    Args:
        folder: 文件夹。
        no_mode: 文件名是否有0000前缀。
        sort: 是否排序进行。
    Returns:
        文件名迭代。
    """
    filenames = os.listdir(folder)
    if sort:
        filenames = sorted(filenames)
    if return_path:
        return [os.path.join(folder, filename) for filename in filenames]
    return filenames


def replace_filename(filename, replacement):
    return replacement + "_" + filename.split('_')[-1]

def copy_properties(new_image, old_image):
    new_image.SetSpacing(old_image.GetSpacing())
    new_image.SetOrigin(old_image.GetOrigin())
    new_image.SetDirection(old_image.GetDirection())


def read_image(path, name):
    file_path = os.path.join(path, name)
    if os.path.isfile(file_path):
        image = silk.ReadImage(file_path)
        array = silk.GetArrayFromImage(image)  # numpy
        array = array.transpose([1, 2, 0])
        return image, array

    return None, None


def read_image_new(path, name=None) -> tuple[Image, ndarray, ndarray[Any, dtype[Any]]]:
    file_path = path if name is None else os.path.join(path, name)
    if os.path.isfile(file_path):
        image = silk.ReadImage(file_path)
        array = silk.GetArrayFromImage(image)  # numpy
        array = array.transpose([2, 1, 0])
        return image, array, np.array([image.GetSpacing()[0], image.GetSpacing()[1], image.GetSpacing()[2]]).astype(
            float)

    print("Can not read missing file {} in {}".format(name, path))
    raise FileNotFoundError


def save_data_image(new_image, path):
    if len(new_image.shape) == 2:
        new_image = np.expand_dims(new_image, axis=2)
    image = silk.GetImageFromArray(new_image.transpose([2, 0, 1]), isVector=False)
    silk.WriteImage(image, wrap_niigz(path))


def save_data_image_new(new_image, path, name, spacing=None):
    maybe_mkdir(path)
    save_data_new(new_image, os.path.join(path, name), spacing=spacing)


def save_data_new(new_image, path, spacing=None):
    image = silk.GetImageFromArray(new_image.transpose([2, 1, 0]), isVector=False)
    if spacing is not None:
        image.SetSpacing(spacing)
    silk.WriteImage(image, wrap_niigz(path))


def save_image(new_image, old_image, filename, name, path=None):
    if path is not None:
        maybe_mkdir(path)
    new_name = replace_filename(filename, name)
    save_image2(new_image, old_image, new_name, path)


def save_image2(new_image, old_image, filename, path=None):
    if path is not None:
        maybe_mkdir(path)
    image = silk.GetImageFromArray(new_image.transpose([2, 0, 1]), isVector=False)
    copy_properties(image, old_image)
    silk.WriteImage(image, os.path.join(path, filename) if path is not None else filename)


def save_image_new(new_image, old_image, path, filename, name=None, overwrite=True, spacing=None):
    if name is not None:
        filename = replace_filename(filename, name)
    maybe_mkdir(path)
    save_path = os.path.join(path, filename)
    if overwrite or not os.path.isfile(save_path):
        image = silk.GetImageFromArray(new_image.transpose([2, 1, 0]), isVector=False)
        copy_properties(image, old_image)
        if spacing is not None:
            image.SetSpacing(spacing)
        silk.WriteImage(image, save_path)


def save_image_raw(new_image, old_image, name):
    image = silk.GetImageFromArray(new_image.transpose([2, 0, 1]), isVector=False)
    copy_properties(image, old_image)
    silk.WriteImage(image, wrap_niigz(name))

def maybe_mkdir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def is_niigz(name):
    return name.endswith(".nii.gz")


def wrap_niigz(name):
    return name + ".nii.gz"


def unwrap_niigz(name):
    return name[:-7]


def wrap_npy(name):
    return name + ".npy"


def unwrap_npy(name):
    return name[:-4]


def wrap_png(name):
    return name + ".png"


def unwrap_png(name):
    return name[:-4]


def wrap_txt(name):
    return wrap_extension(name, "txt")


def unwrap_txt(name):
    return unwrap_extension(name, "txt")


def wrap_mp4(name):
    return wrap_extension(name, "mp4")


def unwrap_mp4(name):
    return unwrap_extension(name, "mp4")


def wrap_extension(name, ext):
    return name + "." + ext


def unwrap_extension(name, ext):
    return name[:-len(ext) - 1]


def append_name(filename, name):
    return wrap_niigz(unwrap_niigz(filename) + name)