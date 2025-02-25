import pandas as pd
from fast_transformers.masking import LengthMask as LM
from view.draw import plot_ccs_comparison,plot_relative_error_boxplot

if __name__ == '__main__':

    test_name = 'test_2'
    # 加载对比的 DataFrame
    
    results_df = pd.read_csv('results/predictions_0.895_attention.csv')  # 需要比较的文件
    # results_df = pd.read_csv('predictions.csv')  # 需要比较的文件

    try:
        plot_ccs_comparison(results_df,f'results_{test_name}.png')
    except Exception as e:
        print(f"error : {e}")

    # dataset = ['./results/predictions_0.8824_best.csv','results/predictions_0.7848181739199651.csv','results/predictions_0.8654300104795007.csv']
    # dataset_lable = ['HyperCCS','CCSP2.0','CCSBase','DeepCCS']
    # colors = ['#5391f5', '#8635a9', '#8e2fa4', '#aa2195', '#d2196b', '#da1653', '#FF1653']

    # plot_relative_error_boxplot(dataset,dataset_lable,colors)