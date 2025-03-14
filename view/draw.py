import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# 读取数据函数
def read_data(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("file not support ,please give xlsx or csv")
    return data

def calculate_statistics_torch(predicted, true):
    # 使用 PyTorch 计算统计量
    diff = predicted - true
    rmse = torch.sqrt(torch.mean(diff ** 2))

    ss_total = torch.sum((true - torch.mean(true)) ** 2)
    ss_residual = torch.sum(diff ** 2)
    r2 = 1 - ss_residual / ss_total

    median_error = torch.median(torch.abs((predicted - true) / true) * 100).item()

    return rmse.item(), r2.item(), median_error

# 计算统计指标
def calculate_statistics(predicted, true):
    rmse = np.sqrt(mean_squared_error(true, predicted))  # 均方根误差

    r2 = r2_score(true, predicted)  # 决定系数

    median_error = np.median(np.abs((predicted - true) / true) * 100)  # 中位相对误差（百分比）
    
    return rmse, r2, median_error


def plot_ccs_comparison_torch(predicted_ccs,true_ccs, fig_name):

    # 计算相对误差(RE)
    relative_error = torch.abs((predicted_ccs - true_ccs) / true_ccs) * 100  # 转为百分比

    # 计算统计指标
    rmse, r2, median_error = calculate_statistics(predicted_ccs, true_ccs)

    print(rmse, r2, median_error)

    # 创建颜色映射
    norm = Normalize(vmin=0, vmax=30)  # 设置颜色映射范围
    cmap = cm.viridis  # 使用viridis色图
    colors = cmap(norm(relative_error.numpy()))

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(predicted_ccs.numpy(), true_ccs.numpy(), c=relative_error.numpy(), cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(scatter, label='Relative Error (RE, %)')

    # 添加y=x参考线
    x = torch.linspace(predicted_ccs.min(), predicted_ccs.max(), 500)
    plt.plot(x.numpy(), x.numpy(), 'k--', label='y = x')

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


# 绘图函数
def plot_ccs_comparison(data,fig_name):

    # 提取数据
    predicted_ccs = data['predicted_ccs'].values
    true_ccs = data['true_ccs'].values
    
    # 计算相对误差(RE)
    relative_error = np.abs((predicted_ccs - true_ccs) / true_ccs) * 100  # 转为百分比

    # 计算统计指标
    rmse, r2, median_error = calculate_statistics(predicted_ccs, true_ccs)

    # #518ef4  #df366b
    # 创建颜色映射
    colors = ['#5391f5','#8635a9','#8e2fa4','#aa2195','#d2196b','#da1653','#FF1653']  # 可根据需要调整颜色值
    cmap = LinearSegmentedColormap.from_list("shap_custom", colors)

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(predicted_ccs, true_ccs, c=relative_error, cmap=cmap, s=10, alpha=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(scatter,cax=cax)
    cbar.set_label('Relative Error (RE, %)', fontsize=14)  # 颜色条标题字体大小
    cbar.ax.tick_params(labelsize=12)  # 调整 colorbar 刻度字体大小

    # 添加y=x参考线
    x = np.linspace(min(predicted_ccs), max(predicted_ccs), 500)
    ax.plot(x, x, 'k--', label='y = x')
    # plt.legend(loc='lower right')  # 将图例放到右下角

    # 标注统计信息
    stats_text = (
        f"Median Error = {median_error:.1f} %\n"
        f"RMSE = {rmse:.3f} Å$^2$\n"
        f"$R^2$ = {r2:.3f}"
    )
    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes, 
            fontsize=14, 
            verticalalignment='top', 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,pad=0.3),
            linespacing=1.8,
            multialignment='left')
    
    min_val = min(min(predicted_ccs), min(true_ccs))  # 数据中的最小值
    max_val = max(max(predicted_ccs), max(true_ccs))  # 数据中的最大值
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # 图形设置y
    ax.set_xlabel(r'Predicted CCS (Å$^2$)', fontsize=15)  # 使用原始 Å 加上标 2
    ax.set_ylabel(r'Measured CCS (Å$^2$)', fontsize=15)
    ax.set_title('CCS Comparison', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)  # 统一调整 x 轴和 y 轴刻度字体大小
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"./{fig_name}")
    plt.show()

    
def plot_relative_error_boxplot(datasets, dataset_labels, colors, 
                                title='relative_error_boxplot', xlabel='models', ylabel='relative_error (%)', 
                                figsize=(10, 10), fontsize_title=14, fontsize_label=12):
    """
    绘制多个数据集的相对误差箱式图，每个数据集使用指定的颜色。

    参数:
    - datasets: list of str, 数据集 CSV 文件路径的列表。
    - dataset_labels: list of str, 每个数据集的标签列表。
    - colors: list of str, 每个数据集的颜色（十六进制格式）。
    - title: str, 图表标题（默认: '各数据集相对误差箱式图'）。
    - xlabel: str, x轴标签（默认: '数据集'）。
    - ylabel: str, y轴标签（默认: '相对误差 (%)'）。
    - figsize: tuple, 图表大小（默认: (10, 6)）。
    - fontsize_title: int, 标题字体大小（默认: 14）。
    - fontsize_label: int, 标签字体大小（默认: 12）。

    返回:
    - None, 直接显示箱式图。
    """

    if len(datasets) > len(colors):
        raise ValueError("颜色列表不足以覆盖所有数据集。")
    if len(datasets) > 7:
        raise ValueError("数据集数量超过最大限制7个。")

    # 读取所有数据集，计算相对误差，并合并到一个 DataFrame 中
    all_data = []
    for dataset, label in zip(datasets, dataset_labels):
        df = pd.read_csv(dataset)
        # 计算相对误差：|true_ccs - predicted_ccs| / true_ccs * 100%
        df['relative_error'] = abs(df['true_ccs'] - df['predicted_ccs']) / df['true_ccs'] * 100
        df['dataset'] = label
        df = df[df['relative_error'] <= 10]
        all_data.append(df)
    all_data = pd.concat(all_data)

    

    # 创建颜色字典，为每个数据集分配颜色
    color_dict = {label: color for label, color in zip(dataset_labels, colors)}

    # 绘制箱式图
    plt.figure(figsize=figsize)
    ax = sns.boxplot(x='dataset', y='relative_error', data=all_data, palette=color_dict,showfliers=False)  # 关键参数：禁用离群值显示)
    plt.title(title, fontsize=fontsize_title)
    plt.xlabel(xlabel, fontsize=fontsize_label)
    plt.ylabel(ylabel, fontsize=fontsize_label)
    # 设置纵轴范围
    plt.ylim(0, 10)  # 新增代码
    plt.xticks(rotation=45)  # 旋转横轴标签，避免重叠

    plt.tight_layout()
    plt.savefig(f"./results/box.png")