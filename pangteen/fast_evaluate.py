import concurrent
import multiprocessing
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from queue import Queue
from time import time

import math
from typing import Optional

import numpy as np
import pandas as pd

from nnunetv2.utilities.file_path_utilities import get_specific_folder, get_output_folder
from pangteen import config, utils
from pangteen.util import metrics as m
from pangteen.util.metrics import COMMON_METRICS, get_worst_val, METRIC_DIRECTIONS


class MetricManager:

    def __init__(self, result_folder, task_config: config.BaseConfig, val=False, train=False, trainer=None):
        """
        Args:
            result_folder: 预测结果文件夹。
            task_config: 任务配置。
        """
        self.labels = list(task_config.label_map.values())[1:]  # 除了背景外的标签。
        self.task_config = task_config
        self.val = val
        self.train = train
        self.trainer = trainer
        self.label_map = {}
        for k, v in task_config.label_map.items():
            self.label_map[v] = k  # 标签名字 -> 标签值。
        self.summary_row_titles = self.labels + ["平均", "最佳", "最差", "75分位", "25分位"]
        self.summary_col_titles = list(COMMON_METRICS.keys())
        self.metric_funcs = COMMON_METRICS.items()
        self.metric_count = len(self.metric_funcs)
        self.summary_rows = len(self.summary_row_titles)
        self.round_limit = 3

        self.predict_result_folder = result_folder

    def evaluate(self):
        """
        逐个遍历待评估的测试数据，计算各个评价指标。
        """
        filenames = os.listdir(self.predict_result_folder)
        filenames = sorted(filenames)

        print("Collecting all test data from {} ...".format(self.predict_result_folder))
        infos = []
        for filename in filenames:
            # 排除其他文件的干扰。
            if not utils.is_niigz(filename):
                continue
            case_id = int(filename.split('.')[0].split('_')[-1])
            if case_id in self.task_config.black_list:
                continue

            infos.append(ImageInfo(filename))
            # if len(infos) > 8:
            #     break

        # 开启多线程计算评价指标。
        start_time = time()
        print("Start parallel evaluating all test data ...")
        with ThreadPoolExecutor(max_workers=config.max_thread_cnt) as executor:
            for info in infos:
                executor.submit(info.evaluate, self)

        print("=========> Start to analyze the result ...")
        self.analyze_result(infos)

        print("=========> Finish evaluating all {} results ! Cost {} seconds.".format(len(infos), time() - start_time))

    def analyze_result(self, infos):
        """
        results: [filename, metric] 第一维是测试数据名，第二维是对应指标平均评估结果。
        """
        # (label, metric)
        summary_table_data = np.zeros((len(self.summary_row_titles), len(self.summary_col_titles)))
        label_count = len(self.labels)
        # (filename, metric)，取标签的平均值。
        metric_results = np.zeros((len(infos), self.metric_count))
        evaluate_name_list = []
        for i, info in enumerate(infos):
            # (label, metric)
            results = info.evaluate_table_data
            evaluate_name_list.append(info.filename)
            summary_table_data += results
            for j in range(self.metric_count):
                metric_results[i, j] = np.mean(results[:label_count, j])

        for i, metric in enumerate(self.summary_col_titles):
            increasing = METRIC_DIRECTIONS.get(metric)

            summary_table_data[-5, i] = sum(summary_table_data[:label_count, i]) / label_count
            summary_table_data[-4, i] = np.max(metric_results[:, i])
            summary_table_data[-3, i] = np.min(metric_results[:, i])
            summary_table_data[-2, i] = np.percentile(metric_results[:, i], 75)
            summary_table_data[-1, i] = np.percentile(metric_results[:, i], 25)
            if not increasing:
                summary_table_data[-4, i], summary_table_data[-3, i] = summary_table_data[-3, i], summary_table_data[-4, i]
                summary_table_data[-2, i], summary_table_data[-1, i] = summary_table_data[-1, i], summary_table_data[-2, i]

        for i, label in enumerate(self.summary_row_titles):
            for j, metric in enumerate(self.summary_col_titles):
                if i <= label_count:
                    summary_table_data[i, j] = summary_table_data[i, j] / len(infos)

                summary_table_data[i, j] = round(summary_table_data[i, j], self.round_limit)
                print("Table data for row : {}; col : {} is {}".format(label, metric, summary_table_data[i, j]))

        table_name = 'evaluate_val_data.xlsx' if self.val else 'evaluate_test_data.xlsx' if not self.train else 'evaluate_train_data.xlsx'
        if self.trainer:
            table_name = self.trainer + '_' + table_name
        table_writer = pd.ExcelWriter(table_name)
        table = pd.DataFrame(data=summary_table_data, index=self.summary_row_titles, columns=self.summary_col_titles)
        table.to_excel(table_writer, sheet_name='Summary')

        for j, (name, metric) in enumerate(self.metric_funcs):
            detail_data = np.zeros((len(evaluate_name_list), len(self.labels)))
            for i, info in enumerate(infos):
                detail_data[i, :] += info.evaluate_table_data[:label_count, j]

            detail_data = np.round(detail_data, self.round_limit)
            metric_table = pd.DataFrame(data=detail_data, index=evaluate_name_list, columns=list(self.labels))
            metric_table.to_excel(table_writer, sheet_name=name)

        table_writer.close()

    def get_label_path(self, image_filename):
        return self.task_config.get_label_path(image_filename)


class ImageInfo:
    def __init__(self, filename):
        self.filename = filename
        self.confusion_matrix = m.ConfusionMatrix()
        self.evaluate_table_data = None

    def evaluate(self, manager):
        """
        评估一张图像的预测结果。
        """
        print("Evaluating {} ...".format(self.filename))
        evaluate_start_time = time()
        predict_image, predict, _ = utils.read_image(manager.predict_result_folder, self.filename)
        gt_image, gt, _ = utils.read_image(manager.get_label_path(self.filename))
        # 冗余了几行，方便计算。
        self.evaluate_table_data = np.zeros((manager.summary_rows, manager.metric_count))
        # 计算每个评价指标的结果。
        for j, (name, metric) in enumerate(manager.metric_funcs):
            start_time = time()
            tmp_val = []
            res = 0
            # 计算每个标签的评价指标。
            for i, label_name in enumerate(manager.labels):
                label = manager.label_map[label_name]
                # 把label变成np.uint8类型，否则会出现错误。
                label = np.uint8(label)
                self.confusion_matrix.set_test(predict == label)
                self.confusion_matrix.set_reference(gt == label)
                metric_val = metric(confusion_matrix=self.confusion_matrix, nan_for_nonexisting=True)
                if math.isnan(metric_val):
                    fix_metric_val = get_worst_val(name)
                    metric_val = fix_metric_val
                    print("Why metric value of {} is NaN ?????????? !, fix it to {}".format(self.filename,
                                                                                            fix_metric_val))

                tmp_val.append(round(metric_val, manager.round_limit))
                self.evaluate_table_data[i, j] = metric_val

            print("cost {} seconds on calculating {} : {}".format(time() - start_time, name, tmp_val))

        print("Evaluating {} done ! Cost {} seconds.".format(self.filename, time() - evaluate_start_time))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to evaluate the test data.')
    parser.add_argument('-i', type=str, required=False, help="The predict result folder to evaluate")
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-t', type=str, required=False,
                        help='The test data name to evaluate, default is default config.')
    parser.add_argument('--evaluate_val', action='store_true', required=False, default=False,
                        help='Evaluate the validation data instead of test data.')
    parser.add_argument('--evaluate_train', action='store_true', required=False, default=False,
                        help='Evaluate the train data instead of test data.')
    args = parser.parse_args()

    folds = [i if i == 'all' else int(i) for i in args.f]
    if args.i is None:
        if args.evaluate_val:
            folds = folds[0]
            result_folder = get_output_folder(args.d, args.tr, args.p, args.c, folds)
            result_folder = os.path.join(result_folder, 'validation')
        else:
            if args.evaluate_train:
                result_folder = get_specific_folder(config.predict_train_folder, args.d, args.tr, args.p, args.c, folds)
            else:
                result_folder = get_specific_folder(config.predict_folder, args.d, args.tr, args.p, args.c, folds)
    else:
        result_folder = args.i

    task_config = utils.get_task_config(args.t)

    manager = MetricManager(result_folder=result_folder, task_config=task_config, val=args.evaluate_val, train=args.evaluate_train, trainer=args.tr)
    manager.evaluate()


if __name__ == '__main__':
    main()
