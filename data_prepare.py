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

    def dataset_separator(df,final_data_name,type):
            
            dataset_H_plus = df[df['Adduct']== '[M+H]+']
            dataset_Na = df[df['Adduct']== '[M+Na]+']
            dataset_H_miner = df[df['Adduct']== '[M-H]-']

            dataset_H_plus.to_excel(target_dir + '/'+ final_data_name +'_H+_'+type+'.xlsx', index=False)
            dataset_Na.to_excel(target_dir + '/'+ final_data_name +'_Na+_'+type+'.xlsx', index=False)
            dataset_H_miner.to_excel(target_dir + '/'+ final_data_name +'_H-_'+type+'.xlsx', index=False)


    for i in range(10,11):

        final_data_name = 'FD_A' + str(i) # Final Data Metlin
        name_flod = '/'+ final_data_name
        target_dir = FINAL_DATA_PATH + name_flod

        # make dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # final_data = Data_reader_ALLCCS([ORINGINAL_BASE_PATH+'/MZ_ISO_METLIN.csv'],
        #                                 ['Molecule Name','CCS_AVG','Adduct','inchi','smiles','m/z'],
        #                                 fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))

        
        final_data = Data_reader_ALLCCS([ORINGINAL_BASE_PATH+'/ALLCCS_ISO_WH.csv'],
                                ['AllCCS ID','Name','Structure','CCS','Adduct','m/z','Formula'],
                                fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))
        print("index is here:")
        print(final_data.data.index)
        final_data.data.rename(columns={'Structure':'smiles'},inplace=True)
        final_data.data['Input'] = final_data.data['Name']

        # final_data.data.rename(columns={'CCS_AVG':'CCS',"Molecule Name":"Name"},inplace=True)
        # final_data.data['Input'] = final_data.data['Name']
        final_data.data.info()
        
        final_data.print_uniques()
        final_data.select_adduct_fre()
        final_data.data.info()

        final_data.data.to_csv('Selected_FullALLCCS2.csv', index=False)

        # write data as csv
        # test_dataset, valid_dataset = final_data.random_split(
        #     [int(len(final_data.data) * 0.2), 
        #     int(len(final_data.data) * 0.1)])
        
        
        # Set dataset name
        # final_data.data.to_csv(target_dir + '/'+ final_data_name +'_train.csv', index=False)
        # dataset_separator(final_data.data,final_data_name,type='train')

        # test_dataset.to_csv(target_dir +'/'+ final_data_name +'_test.csv' , index=False)
        # dataset_separator(test_dataset,final_data_name,type='test')

        # valid_dataset.to_csv(target_dir +'/'+ final_data_name +'_valid.csv' , index=False)
        # dataset_separator(valid_dataset,final_data_name,type='valid')