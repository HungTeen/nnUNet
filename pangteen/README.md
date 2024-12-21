数据集格式转换指令如下：
```bash
python -u pangteen/dataset/Dataset100_Ablation.py data label -d 101
python -u pangteen/dataset/Dataset100_Ablation.py contrast label -d 102
```

数据集预处理指令如下：
```bash
nnUNetv2_plan_and_preprocess -d 101 102 -c 3d_fullres --verify_dataset_integrity
```

模型训练代码如下：
```bash
CUDA_VISIBLE_DEVICES=2 nohup nnUNetv2_train 101 3d_fullres 0 -num_gpus 1 > main.out 2>&1 &

```

模型预测代码如下：
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/predict.py -d 101 -c 3d_fullres -f 0 --disable_tta > main2.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/predict.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d 101 -c 3d_fullres -f 0 > main2.out 2>&1 &
```

评估代码如下：
```bash
nohup python -u pangteen/evaluate.py label -d 101 -c 3d_fullres -f 0 > main2.out 2>&1 &
```

时间分析代码如下：
```bash
python -u pangteen/time_analyze.py main2.out
```