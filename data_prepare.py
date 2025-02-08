import os
import pandas as pd
from data_pre import ALLCCS_PATH,ALLCCS_EXP_PATH,RESULTS_BASE_PATH
from data_pre.data_module import Data_reader_METLIN,Data_reader_ALLCCS

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", 1000)  # 调整宽度，避免换行
pd.set_option("display.max_colwidth", None) 

if __name__ == '__main__':
    # Allccs data 
    allccs = Data_reader_ALLCCS(ALLCCS_PATH,
                                fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))

    allccs.selected_proprties({'Type': 'Experimental CCS'})


    allccs_exp = Data_reader_ALLCCS(ALLCCS_EXP_PATH,
                                    target_colnames=["Adduct","m/z","CCS","Type","Approach","ID"],
                                    fun=lambda path, col_name: pd.read_csv(path, usecols=col_name))

    allccs_exp.selected_proprties({'Type': 'TIMS',"Approach":"Empirical method"})

    # merge allccs data by ID
    merged_df = pd.merge(allccs_exp.data,allccs.data,left_on="ID",right_on="AllCCS ID", how="inner")

    #select Adduct equal 
    merged_df = merged_df[merged_df["Adduct_x"] == merged_df["Adduct_y"]]

    #select ccs equal 
    merged_df = merged_df[merged_df["CCS_x"].astype(int) == merged_df["CCS_y"].astype(int)]

    #select col name
    merged_df = merged_df[['AllCCS ID','Adduct_x','CCS_x','m/z_x','Structure','Formula']]

    # fix col name 
    merged_df = merged_df.rename(columns={
    "Adduct_x": "Adduct",
    "m/z_x": "m/z",
    "CCS_x": "CCS"})

    # select data with adduct type
    desired_adducts = ["[M+H]+", "[M+Na]+", "[M-H]-"]

    # 针对合并后的 DataFrame 筛选出满足条件的行
    merged_df = merged_df[merged_df["Adduct"].isin(desired_adducts)]

    # 查看结果
    merged_df.info()
    merged_df.to_csv(RESULTS_BASE_PATH + '/ALLCCS_FINAL.csv',index=False)
    print(merged_df.head(20))



