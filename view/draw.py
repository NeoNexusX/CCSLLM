import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# 读取数据函数
def read_data(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("file not support ,please give xlsx or csv")
    return data


# 计算统计指标
def calculate_statistics(predicted, true):
    rmse = np.sqrt(mean_squared_error(true, predicted))  # 均方根误差
    r2 = r2_score(true, predicted)  # 决定系数
    median_error = np.median(np.abs((predicted - true) / true) * 100)  # 中位相对误差（百分比）
    return rmse, r2, median_error


# 绘图函数
def plot_ccs_comparison(data,fig_name):
    # 提取数据
    predicted_ccs = data['predicted_ccs']
    true_ccs = data['true_ccs']
    
    # 计算相对误差(RE)
    relative_error = np.abs((predicted_ccs - true_ccs) / true_ccs) * 100  # 转为百分比

    # 计算统计指标
    rmse, r2, median_error = calculate_statistics(predicted_ccs, true_ccs)

    # 创建颜色映射
    norm = Normalize(vmin=0, vmax=30)  # 设置颜色映射范围
    cmap = cm.viridis  # 使用viridis色图
    colors = cmap(norm(relative_error))

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(predicted_ccs, true_ccs, c=relative_error, cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(scatter, label='Relative Error (RE, %)')

    # 添加y=x参考线
    x = np.linspace(min(predicted_ccs), max(predicted_ccs), 500)
    plt.plot(x, x, 'k--', label='y = x')

    # 标注统计信息
    stats_text = (
        f"Median Error = {median_error:.1f} %\n"
        f"RMSE = {rmse:.3f} Å²\n"
        f"$R^2$ = {r2:.3f}"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 图形设置
    plt.xlabel('Predicted CCS (Å²)')
    plt.ylabel('Measured CCS (Å²)')
    plt.title('CCS Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
