import pandas as pd


def analyze(path='main2.out'):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line for line in lines if line.startswith('Done predicting for ')]
        table = []
        rows = []
        for line in lines:
            parts = line.split('seconds')
            line = parts[0]
            elements = line.split(' ')
            filename = elements[-4].split('/')[-1]
            predict_time = round(float(elements[-2]), 2)
            table.append([predict_time])
            rows.append(filename)
            print(f'{filename}: {predict_time}s')

        table = pd.DataFrame(table, index=rows, columns=['耗时（秒）'])
        # 保存到excel。
        table.to_excel('predict_time.xlsx')
        print(len(lines))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to analyze the time of predicting.')
    parser.add_argument('file', type=str, help="The output file to analyze")
    args = parser.parse_args()
    analyze(args.file)


if __name__ == '__main__':
    main()