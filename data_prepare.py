import os
import pandas as pd
from data_pre import RESULTS_BASE_PATH,Final_MELTIN_PATH,FINAL_DATA_PATH,ORINGINAL_BASE_PATH
from data_pre.data_module import Data_reader_METLIN,Data_reader_ALLCCS

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", 1000)  # 调整宽度，避免换行
pd.set_option("display.max_colwidth", None) 

if __name__ == '__main__':

    #dataset maybe : 7:2:1
    #dataset maybe : 8:1:1
    #m/z / 1000
    #devide into 5 parts:

    for i in range(5, 10):

        name_flod = f"/{i}"
        target_dir = FINAL_DATA_PATH + name_flod

        # make dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        final_data = Data_reader_ALLCCS([ORINGINAL_BASE_PATH+'/final_data.csv'],
                                        ['CCS','m/z','Adduct','smiles'],
                                        fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
        final_data.data.info()

        final_data.data['Compound Name'] = final_data.data['smiles']
        final_data.data['Input'] = final_data.data['smiles']

        # write data as csv
        test_dataset, valid_dataset = final_data.random_split(
            [int(len(final_data.data) * 0.2), int(len(final_data.data) * 0.1)])
        
        final_data_name = 'FINAL_DATA'
        
        # Set dataset name
        final_data.data.to_csv(target_dir + '/'+ final_data_name +'_train.csv', index=False)

        train_H_plus = final_data.data[final_data.data['Adduct']== '[M+H]+']
        train_Na = final_data.data[final_data.data['Adduct']== '[M+Na]+']
        train_H_miner = final_data.data[final_data.data['Adduct']== '[M-H]-']

        train_H_plus.to_excel(target_dir + '/'+ final_data_name +'_H+_'+'_train.xlsx', index=False)
        train_Na.to_excel(target_dir + '/'+ final_data_name +'_Na+_'+'_train.xlsx', index=False)
        train_H_miner.to_excel(target_dir + '/'+ final_data_name +'_H-_'+'_train.xlsx', index=False)

        test_dataset.to_csv(target_dir + '/'+ final_data_name +'_test.csv', index=False)

        test_H_plus = test_dataset[test_dataset['Adduct'] ==  '[M+H]+']
        test_Na = test_dataset[test_dataset['Adduct'] ==  '[M+Na]+']
        test_H_miner = test_dataset[test_dataset['Adduct'] ==  '[M-H]-']

        test_H_plus.to_excel(target_dir + '/'+ final_data_name +'_H+_'+'_test.xlsx', index=False)
        test_Na.to_excel(target_dir + '/'+ final_data_name +'_Na+_'+'_test.xlsx', index=False)
        test_H_miner.to_excel(target_dir + '/'+ final_data_name +'_H-_'+'_test.xlsx', index=False)


        valid_dataset.to_csv(target_dir +'/'+ final_data_name +'_valid.csv' , index=False)

        valid_H_plus = valid_dataset[valid_dataset['Adduct'] ==  '[M+H]+']
        valid_Na = valid_dataset[valid_dataset['Adduct'] ==  '[M+Na]+']
        valid_H_miner = valid_dataset[valid_dataset['Adduct'] ==  '[M-H]-']

        valid_H_plus.to_excel(target_dir + '/'+ final_data_name +'_H+_'+'_valid.xlsx', index=False)
        valid_Na.to_excel(target_dir + '/'+ final_data_name +'_Na+_'+'_valid.xlsx', index=False)
        valid_H_miner.to_excel(target_dir + '/'+ final_data_name +'_H-_'+'_valid.xlsx', index=False)