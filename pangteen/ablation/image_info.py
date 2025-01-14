import argparse
import os.path

from pangteen import config, utils

if __name__ == '__main__':
    '''
    python pangteen/image_info.py Ablation/origin/data
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument('image', type=str)
    args = parse.parse_args()
    folder = os.path.join(config.my_folder, args.image)
    for filename in utils.next_file(folder):
        image, array, spacing = utils.read_image(folder, filename)
        print("Image: {}, shape: {}, spacing: {}".format(filename, array.shape, spacing))
        x, y, z = array.shape[0] * spacing[0], array.shape[1] * spacing[1], array.shape[2] * spacing[2]
        x = round(x, 2)
        y = round(y, 2)
        z = round(z, 2)
        print("Real Size: ({}, {}, {}) -> ({}, {}, {})".format(array.shape[0], array.shape[1], array.shape[2], x, y, z / 2.5))