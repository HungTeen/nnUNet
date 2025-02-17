import os
import random
import sys
from time import time

import math
import numpy as np
import pandas as pd

from nnunetv2.utilities.file_path_utilities import get_specific_folder, get_output_folder
from pangteen import config, utils
from pangteen.util import metrics as m

COMMON_METRICS = {
    # "False Positive Rate": m.false_positive_rate,
    "Dice": m.dice,  # 重叠率，越大越好。
    "IOU": m.jaccard,  # 交并比，另一种重叠率，越大越好（IOU）。
    # "HD": m.hausdorff_distance,  # 豪斯多夫距离，越小越好。Dice对mask的内部填充比较敏感，而hausdorff distance 对分割出的边界比较敏感。
    "HD95": m.hausdorff_distance_95,  # 豪斯多夫距离去掉最大的一些值，越小越好。
    "Precision": m.precision,  # 精确率，越大越好。
    "Recall": m.recall,  # 查全率，越大越好。
    # "ASSD": m.avg_surface_distance_symmetric,  # 双边平均距离，越小越好。
    # "Avg. Surface Distance": m.avg_surface_distance, # 单边平均距离，越小越好。
    # "Accuracy": m.accuracy, # 预测类别正确的像素数占总像素数的比例，越大越好。 【没什么参考性，都是98%】
    # "F1 Score": (m.fscore, True), # 综合考虑精确率和查全率，越大越好。
    # 当Precision精确率更重要些，就调整β的值小于1
    # 当Precision精确率和Recall召回率的重要性相同，β=1时，称为F1-score，权重相同
    # 当Recall召回率更重要些，就调整β的值大于1
    # "False Omission Rate": m.false_omission_rate,
    # "Negative Predictive Value": m.negative_predictive_value,
    # "False Negative Rate": m.false_negative_rate,
    # "True Negative Rate": m.true_negative_rate,
    # "False Discovery Rate": m.false_discovery_rate,
    # "Total Positives Test": m.total_positives_test,
    # "Total Negatives Test": m.total_negatives_test,
    # "Total Positives Reference": m.total_positives_reference,
    # "total Negatives Reference": m.total_negatives_reference
}

METRIC_DIRECTIONS = {
    "Dice": True,
    "IOU": True,
    "HD": False,
    "HD95": False,
    "Precision": True,
    "Recall": True,
    "ASSD": False,
}


def increasing_metric(name):
    return METRIC_DIRECTIONS.get(name)


def get_worst_val(name):
    return 0 if increasing_metric(name) else 1e3


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

        self.summary_table_data = np.zeros((len(self.summary_row_titles), len(self.summary_col_titles)))
        self.average_data = []  # 数据 -> 评价指标 -> 平均结果。
        self.detail_table_data = {}  # 评价指标 -> 数据 -> 五个段的评估结果。
        self.test_data_name = []  # 测试集名字。
        self.predict_result_folder = result_folder


    def evaluate(self):
        """
        逐个遍历待评估的测试数据，计算各个评价指标。
        """
        filenames = os.listdir(self.predict_result_folder)
        filenames = sorted(filenames)
        for filename in filenames:
            # 排除其他文件的干扰。
            if not utils.is_niigz(filename):
                continue

            print("=========> start for filename {}".format(filename))
            info = ImageInfo(self, filename)
            start_time = time()

            metric_results = info.evaluate_common_metrics(self)

            self.average_data.append(metric_results)
            self.test_data_name.append(filename)

            end_time = time()
            print("=========> cost {} seconds on evaluating test data".format(end_time - start_time))
            start_time = end_time

        self.postprocess()

        print("=========> Finish evaluating all {} results !".format(len(self.average_data)))

    def postprocess(self):
        """
        results: [filename, metric] 第一维是测试数据名，第二维是对应指标平均评估结果。
        """
        results = np.array(self.average_data)
        size = len(self.labels)
        for i, metric in enumerate(self.summary_col_titles):
            increasing = METRIC_DIRECTIONS.get(metric)

            self.summary_table_data[-5, i] = sum(self.summary_table_data[:size, i]) / size
            self.summary_table_data[-4, i] = np.max(results[:, i])
            self.summary_table_data[-3, i] = np.min(results[:, i])
            self.summary_table_data[-2, i] = np.percentile(results[:, i], 75)
            self.summary_table_data[-1, i] = np.percentile(results[:, i], 25)
            if not increasing:
                self.summary_table_data[-4, i], self.summary_table_data[-3, i] = self.summary_table_data[-3, i], self.summary_table_data[-4, i]
                self.summary_table_data[-2, i], self.summary_table_data[-1, i] = self.summary_table_data[-1, i], self.summary_table_data[-2, i]

        for i, label in enumerate(self.summary_row_titles):
            for j, metric in enumerate(self.summary_col_titles):
                if i <= size:
                    self.summary_table_data[i, j] = self.summary_table_data[i, j] / len(results)

                self.summary_table_data[i, j] = round(self.summary_table_data[i, j], 4)
                print("Table data for row : {}; col : {} is {}".format(label, metric, self.summary_table_data[i, j]))
        table_name = 'evaluate_val_data.xlsx' if self.val else 'evaluate_test_data.xlsx' if not self.train else 'evaluate_train_data.xlsx'
        if self.trainer:
            table_name = self.trainer + '_' + table_name
        table_writer = pd.ExcelWriter(table_name)
        table = pd.DataFrame(data=self.summary_table_data, index=self.summary_row_titles, columns=self.summary_col_titles)
        table.to_excel(table_writer, sheet_name='Summary')

        for j, (name, metric) in enumerate(COMMON_METRICS.items()):
            detail_data = self.detail_table_data.get(name)
            print(len(detail_data), len(detail_data[0]))
            print(len(self.test_data_name))
            cols = ['消融区域体素数/总体素数', ] + list(self.labels)
            metric_table = pd.DataFrame(data=detail_data, index=self.test_data_name, columns=cols)
            metric_table.to_excel(table_writer, sheet_name=name)

        table_writer.close()

    def get_label_path(self, image_filename):
        return self.task_config.get_label_path(image_filename)


class ImageInfo:
    def __init__(self, manager: MetricManager, filename):
        start_time = time()
        self.filename = filename
        self.predict_image, self.predict, spacing1 = utils.read_image(manager.predict_result_folder, filename)
        self.gt_image, self.gt, spacing2 = utils.read_image(manager.get_label_path(filename))
        self.confusion_matrix = m.ConfusionMatrix()
        print("cost {} seconds on initialize image".format(time() - start_time))

    def evaluate_common_metrics(self, manager):
        metric_results = []
        for j, (name, metric) in enumerate(COMMON_METRICS.items()):
            start_time = time()
            tmp_val = []
            res = 0
            for i, label_name in enumerate(manager.labels):
                label = manager.label_map[label_name]
                # 把label变成np.uint8类型，否则会出现错误。
                label = np.uint8(label)
                self.confusion_matrix.set_test(self.predict == label)
                self.confusion_matrix.set_reference(self.gt == label)
                metric_val = metric(confusion_matrix=self.confusion_matrix, nan_for_nonexisting=True)
                if math.isnan(metric_val):
                    fix_metric_val = get_worst_val(name)
                    metric_val = fix_metric_val
                    print("Why metric value of {} is NaN ?????????? !, fix it to {}".format(self.filename,
                                                                                            fix_metric_val))
                res += metric_val
                tmp_val.append(round(metric_val, 2))
                manager.summary_table_data[i, j] += metric_val

            metric_results.append(res / len(manager.labels))
            row_data = ['{}/{}'.format(np.sum(self.gt != 0), self.gt.shape[0] * self.gt.shape[1] * self.gt.shape[2]), ] + tmp_val
            manager.detail_table_data.setdefault(name, []).append(row_data)
            print("cost {} seconds on calculating {} : {}".format(time() - start_time, name, tmp_val))

        return metric_results


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
