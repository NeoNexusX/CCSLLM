import pandas as pd
from fast_transformers.masking import LengthMask as LM
from view.draw import plot_residuals,plot_ccs_comparison,plot_relative_error_boxplot,plot_adduct_analysis,plot_in_house_compare,plot_adduct_violion,plot_mz_ccs_compare,plot_adduct_re

if __name__ == '__main__':

    # test_name = 'predictions_ATT_FULL_A0'
    # # 加载对比的 DataFrame
    
    # results_df = pd.read_csv('results/'+test_name+'.csv')  # 需要比较的文件

    # try:
    #     plot_ccs_comparison(results_df,'results/'+f'results_{test_name}.png')
    # except Exception as e:
    #     print(f"error : {e}")

    # dataset = ['results/predictions_ATT_FULL_M0.csv',
    #            'results/KerasECFP_M0.csv',
    #            'results/ccsp2_METLIN_only_result.csv',
    #            'results/0.8278_molfomer_M0FD_M0_7_1e-05_48_0.0.csv']

    # dataset = ['results/predictions_ATT_FULL_A0.csv',
    #            'results/KerasECFP_A0.csv',
    #            'results/ccsp2_AllCCS2_only_result.csv',
    #            'results/0.9334_molfomer_A0FD_A0_18_1e-05_48_0.1.csv',]

    # dataset_lable = ['HyperCCS','KerasECFP','CCSP2','Molformer']
    # colors = ['#6ac1a5', '#e88d67', '#8ca1c9', '#e790c2', '#d2196b', '#a6d558']
    # name = 'ALLCCS'
    # 获取统计量而不显示中间输出
    # stats = plot_relative_error_boxplot(dataset, dataset_lable, colors)

    # # 后续可以自行处理统计结果
    # print(stats[['Model', 'IQR']])  # 只查看模型和IQR
    # stats.to_csv('iqr_stats.csv')   # 保存统计结果

    data_path = './results/predictions_ATT_FULL_A0.csv'

    # plot_adduct_analysis(data_path,"test")

    # plot_adduct_violion(data_path,"test")


    #####
    # # ALLCCS2 数据集
    # data_allccs2 = {
    #     "Ion": ["[M+Na]+", "[M-H]-", "[M+H]+", "[M+H-2H2O]+", "[M+Na-2H]-", 
    #             "[M-H2O+H]+", "[M-H+2Na]+", "[M-H+COO]-", "[2M+Na]+", "[M+NH4]+"],
    #     "Median Error (%)": [1.7, 1.5, 1.4, 0.6, 2.4, 1.6, 1.1, 0.5, 1.9, 1.2],
    #     "Dataset": "ALLCCS2"
    # }

    # # METLIN 数据集
    # data_metlin = {
    #     "Ion": ["[M+Na]+", "[M-H]-", "[M+H]+"],
    #     "Median Error (%)": [1.4, 1.7, 1.3],
    #     "Dataset": "METLIN"
    # }

    # # 合并为 DataFrame
    # df_allccs2 = pd.DataFrame(data_allccs2)
    # df_metlin = pd.DataFrame(data_metlin)
    # df = pd.concat([df_allccs2, df_metlin])
    # plot_adduct_re(df)

    # plot mz with ccs:
    # metlin_path = "original_data/Selected_FullALLCCS2.csv"
    # plot_mz_ccs_compare(metlin_path,"Selected_FullALLCCS2")

    plot_residuals(data_path,'test')