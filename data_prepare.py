import os
import pandas as pd
from data_pre import BASE_PATH, Final_MELTIN_PATH,TEMP_SAVE
from data_pre.data_module import Data_reader_METLIN

if __name__ == '__main__':
    for i in range(5, 10):
        name_flod = f"/{i}"
        target_dir = TEMP_SAVE + name_flod

        # 创建目标目录
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        # 定义一个函数，用于获取相同 Molecule Name, Adduct, 和 inchi 的 m/z 值

        meltin_tester = Data_reader_METLIN(Final_MELTIN_PATH,
                                           fun=lambda path,
                                           col_name: pd.read_csv(path, usecols=col_name))
        # scaler = StandardScaler()
        # meltin_tester.data['m/z'] = scaler.fit_transform(meltin_tester.data[['m/z']])
        # write data as csv
        test_dataset, valid_dataset = meltin_tester.random_split(
            [int(len(meltin_tester.data) * 0.2), int(len(meltin_tester.data) * 0.1)])

        meltin_tester.data.to_csv(target_dir + '/ISO_METLIN_train.csv', index=False)
        test_dataset.to_csv(target_dir + '/ISO_METLIN_test.csv', index=False)
        valid_dataset.to_csv(target_dir + '/ISO_METLIN_valid.csv', index=False)
