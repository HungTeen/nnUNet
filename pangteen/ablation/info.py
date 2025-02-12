def transform(origin_folder, target_folder, origin_filename, is_seg):
    start_time = time()
    print(f"Transforming {origin_filename} ...")
    # 读取预测的肿瘤图像。
    origin_image, origin_array, origin_spacing = utils.read_image(origin_folder, origin_filename)
    print(origin_array.shape, origin_spacing)
    zSize = int(origin_array.shape[2] * origin_spacing[2])
    new_array = resample(origin_array, (origin_array.shape[0], origin_array.shape[1], zSize), is_seg)
    print(new_array.shape)
    utils.save_image(new_array, origin_image, target_folder, origin_filename, spacing=(origin_spacing[0], origin_spacing[1], 1))
    print(f"Transformed {origin_filename} ! Time cost: {time() - start_time:.2f}s")


if __name__ == '__main__':
    '''
    nohup python -u pangteen/ablation/rescale_image.py > main.out 2>&1 &
    '''
    p = Pool(config.max_cpu_cnt)  # 多进程。
    print("=========> Start transform tasks !")
    start_time = time()
    # origin_folder = config.ablation_config.origin_image_folder  # 原始数据集的图像文件夹。
    # target_folder = config.tumor_ablation_config.rescale_image_folder  # 原始数据集的图像文件夹。
    origin_folder = config.ablation_config.origin_label_folder  # 原始数据集的标签文件夹。
    target_folder = config.tumor_ablation_config.rescale_label_folder  # 原始数据集的标签文件夹。
    is_seg = True

    for filename in utils.next_file(origin_folder, sort=True):
        # transform(folder, filename)
        p.apply_async(transform, args=(origin_folder, target_folder, filename, is_seg))

    p.close()
    p.join()
    print("=========> Finish transform tasks ! Time cost: {:.2f}s".format(time() - start_time))