import os
import pandas as pd
from data_pre import ALLCCS_PATH,ALLCCS_EXP_PATH,RESULTS_BASE_PATH,ALLCCS_FINAL_PATH,Final_MELTIN_PATH
from data_pre.data_module import Data_reader_METLIN,Data_reader_ALLCCS

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", 1000)  # 调整宽度，避免换行
pd.set_option("display.max_colwidth", None) 

if __name__ == '__main__':

    # Allccs data 
    allccs = Data_reader_ALLCCS([RESULTS_BASE_PATH+'/final_data.csv'],
                                ['CCS','m/z','Adduct','smiles'],
                                fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
    allccs.data.info()

    for col in allccs.data.columns:
        print(f'col is {col} \r\n '
                f'numbers: {len(allccs.data[col].unique())}\r\n'
                f'types :{allccs.data[col].unique()}\r\n')
        top_10 = allccs.data[col].value_counts().head(10)  # 获取前10个出现频次最多的值
        print(top_10)
        print('-' * 50)  # 分隔线，方便阅读

