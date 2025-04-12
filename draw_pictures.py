import pandas as pd
from fast_transformers.masking import LengthMask as LM
from view.draw import plot_ccs_comparison,plot_relative_error_boxplot

if __name__ == '__main__':

    test_name = '0.9011_ATT_DelecfpFD_M0_160_1e-05_64_0.0'
    # 加载对比的 DataFrame
    
    results_df = pd.read_csv('results/'+test_name+'.csv')  # 需要比较的文件

    try:
        plot_ccs_comparison(results_df,'results/'+f'results_{test_name}.tif')
    except Exception as e:
        print(f"error : {e}")

    # dataset = ['results/0.9921_ATT_12_ACFD_A0_1e-06_32_0.0.csv',
    #            'results/0.97_MLLinear_A0FD_A0_1e-05_64_0.0.csv',
    #            'results/0.9705_LE_A0.csv',
    #            'results/ccsp2_allccs2.csv',]
    # dataset_lable = ['HyperCCS','Molformer','KerasECFP','CCSP2']
    # colors = ['#6ac1a5', '#e88d67', '#8ca1c9', '#e790c2', '#d2196b', '#a6d558']
    # name = 'METLIN'
    # plot_relative_error_boxplot(dataset,dataset_lable,colors,name)