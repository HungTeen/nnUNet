from os.path import join

import pandas as pd

from pangteen import config

black_list=[45]
excel_filename='origin_dataset.xlsx'
test_set_num=60

def get_valid_ids(threshold=70) -> list :
    excel_folder = config.renal_config.origin_folder
    excel_path = join(excel_folder, excel_filename)
    excel = pd.read_excel(excel_path)
    # 遍历每行，如果列A的值为1，则输出整行。
    id_list = []
    for index, row in excel.iterrows():
        if row['右肾数据质量'] >= threshold and row['左肾数据质量'] >= threshold:
            id_list.append(row['编号'])

    print(len(id_list))
    return id_list

def get_case_id(filename) -> int:
    return int(filename.split('_')[-1].split('.')[0])

if __name__ == '__main__':
    get_valid_ids(70)
