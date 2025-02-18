import pandas as pd
from fast_transformers.masking import LengthMask as LM
from view.draw import plot_ccs_comparison


if __name__ == '__main__':
    test_name = 'test'
    # 加载对比的 DataFrame
    
    results_df = pd.read_csv('./results/test_results.csv')  # 需要比较的文件
    # results_df = pd.read_csv('predictions.csv')  # 需要比较的文件

    try:
        plot_ccs_comparison(results_df,f'results_{test_name}.png')
    except Exception as e:
        print(f"error : {e}")