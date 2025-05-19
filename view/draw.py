import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from scipy import stats

# 全局字体设置（路径方式加载 Arial）
arial_font_path = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
arial_font = fm.FontProperties(fname=arial_font_path,size=25)
fm.fontManager.addfont(arial_font_path)
plt.rcParams['font.family'] = arial_font.get_name()
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体仍使用 STIX（可选）

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


def plot_adduct_analysis(data_path, fig_name):

    data = pd.read_csv(data_path)
    data['Relative Absolute Error'] = np.abs((data['predicted_ccs'] - data['true_ccs']) / data['true_ccs']) * 100
    adducts = data['Adduct'].unique()
    colors= [
    '#f1ca5c',  # 柔和亮黄
    '#d8573c',  # 赤陶橙红（暖色对比）
    '#aa2195',  # 紫红色
    '#c4b7d4',  # 薰衣草灰（紫色延展）
    '#7e9abf',  # 雾霾蓝
    '#5391f5',  # 中亮蓝
    '#3c3f58',  # 深靛蓝灰
    '#415753',  # 深青灰
    '#8aaf9c',  # 柔和灰绿
    '#82bd86'   # 柔和绿色
    ]
    colors= ['#f1ca5c','#7e9abf','#82bd86','#aa2195', '#5391f5']
    adduct_colors = {adduct: colors[i % len(colors)] for i, adduct in enumerate(adducts)}
    median_errors = data.groupby('Adduct')['Relative Absolute Error'].median().sort_values(ascending=False)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)
    
    for adduct in adducts:
        subset = data[data['Adduct'] == adduct]
        ax.scatter(subset['predicted_ccs'], subset['true_ccs'], 
                   c=adduct_colors[adduct], label=adduct, s=20, alpha=0.5)
    
    min_val = min(data['predicted_ccs'].min(), data['true_ccs'].min())
    max_val = max(data['predicted_ccs'].max(), data['true_ccs'].max())
    x = np.linspace(min_val, max_val, 500)
    ax.plot(x, x, 'k--', label='y = x')
    

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # stats_lines = ["Median Error:"]
    # stats_lines.extend(f"{adduct}: {error:.1f}%" for adduct, error in median_errors.items())

    # # 保持第一行不变，其余行按长度降序排序
    # if len(stats_lines) > 1:
    #     stats_lines[1:] = sorted(stats_lines[1:], key=lambda x: -len(x))

    # ax.text(0.02, 0.98, "\n".join(stats_lines),
    #         transform=ax.transAxes, 
    #         fontsize=17,
    #         linespacing=2,
    #         horizontalalignment='left', verticalalignment='top',
    #         bbox=dict(boxstyle="round", facecolor="white", alpha=0.1))
    
    ax.legend(handles=[Line2D([0], [0], 
                              marker='o', 
                              color='w', 
                              label=adduct,
                              markerfacecolor=adduct_colors[adduct], 
                              markersize=9) 
                      for adduct in adducts], loc='lower right', fontsize=18)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    # ax.set_xlabel(r'Predicted CCS (Å$^2$)',fontproperties=arial_font ,fontsize=24,labelpad=15)
    # ax.set_ylabel(r'Measured CCS (Å$^2$)', fontproperties=arial_font, fontsize=24,labelpad=15)
    ax.tick_params(axis='both', labelsize=25)
    ax.set_aspect('equal')
    # plt.tight_layout(pad=3.0) 
    plt.savefig(f"./{fig_name}_scatter.png", dpi=600, bbox_inches='tight')
    plt.close()

    #for bar figure:
    adducts = data['Adduct'].unique()
    colors = ['#D978A8', '#6A8EAD', '#E8A87C']
    
    bins = [0, 1, 3, np.inf]
    labels = ['≤ 1%', '1-3%', '> 3%']
    
    error_counts = []
    for adduct in adducts:
        subset = data[data['Adduct'] == adduct]
        counts = pd.cut(subset['Relative Absolute Error'], bins=bins, labels=labels).value_counts()
        error_counts.append(counts)
    
    error_counts = pd.concat(error_counts, axis=1).T
    error_counts.index = adducts

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    bottom = np.zeros(len(adducts))
    for i, label in enumerate(labels):
        ax.bar(adducts, error_counts[label], bottom=bottom, 
               label=f'ARE {label}', 
               color=colors[i % len(colors)], 
               alpha=0.9, 
               width=0.6)
        bottom += error_counts[label]
    
    ax.tick_params(axis='x', rotation=45,labelsize=25)
    ax.tick_params(axis='y', rotation=45, labelsize=32)
    ax.legend(title='Absolute Relative\n    Error Range', loc='upper right', fontsize=22, title_fontsize=25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_ylabel('Number of Ions', fontsize=30, fontproperties=arial_font,labelpad=18)
    # ax.set_xlabel("Adduct Type", fontsize=30, fontproperties=arial_font,labelpad=18)
    plt.gca().set_xticklabels([]) 
    plt.tight_layout()
    plt.savefig(f"./{fig_name}_bar.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot_ccs_comparison(data, fig_name):
    
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
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(predicted_ccs, true_ccs, c=relative_error, cmap=cmap, s=12, alpha=1,vmin=0, vmax=50)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(scatter,cax=cax,extend='max')
    cbar.set_label('Relative Error (RE, %)', fontproperties=arial_font, size=22)  # 颜色条标题字体大小
    cbar.ax.tick_params(labelsize=20)  # 调整 colorbar 刻度字体大小

    # 添加y=x参考线
    x = np.linspace(min(predicted_ccs), max(predicted_ccs), 500)
    ax.plot(x, x, 'k--', label='y = x')

    
    ax.spines['right'].set_visible(False)  # 移除右边框
    ax.spines['top'].set_visible(False)    # 移除上边框

    # 标注统计信息
    stats_text = (
        f"Median Error = {median_error:.3f} %\n"
        f"RMSE = {rmse:.5f} Å$^2$\n"
        f"$R^2$ = {r2:.5f}"
    )
    ax.text(0.05, 
            0.95, 
            stats_text, 
            transform=ax.transAxes, 
            fontproperties=arial_font, 
            size=20,
            verticalalignment='top', 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,pad=0.3),
            linespacing=2,
            multialignment='left')
    
    min_val = min(min(predicted_ccs), min(true_ccs))  # 数据中的最小值
    max_val = max(max(predicted_ccs), max(true_ccs))  # 数据中的最大值
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # 图形设置y
    # ax.set_xlabel(r'Predicted CCS (Å$^2$)', labelpad=20, fontproperties=arial_font, size=25)  # 使用原始 Å 加上标 2
    # ax.set_ylabel(r'Measured CCS (Å$^2$)', labelpad=20, fontproperties=arial_font, size=25)
    ax.tick_params(axis='both', labelsize=24)  # 统一设置刻度字号
    ax.set_yticks([140, 160, 180, 200,220,240])  # 设置固定的纵轴刻度
    ax.set_xticks([140, 160, 180, 200,220,240])  # 设置固定的纵轴刻度
    # ax.set_yticks([150, 200, 250, 300])  # 设置固定的纵轴刻度
    # ax.set_xticks([150, 200, 250, 300])  # 设置固定的纵轴刻度
    ax.set_aspect('equal', adjustable='box')
    
    # 确保所有文本都使用Arial
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(arial_font)

    plt.tight_layout()
    plt.savefig(f"./{fig_name}",dpi=600)


def plot_adduct_violion(data_path, fig_name):
    # Read data
    df = pd.read_csv(data_path)

    # Calculate relative error (%)
    df['Relative_Error'] = (df['predicted_ccs'] - df['true_ccs']) / df['true_ccs'] * 100

    # Add ion polarity column (positive/negative)
    df['Polarity'] = df['Adduct'].apply(lambda x: 'Positive' if '+' in x else 'Negative')

    # Filter ions with relative error > 15%
    high_error_ions = df[abs(df['Relative_Error']) > 15]

    # Output high error ions' SMILES
    if not high_error_ions.empty:
        print("SMILES of ions with relative error > 15%:")
        print(high_error_ions[['smiles', 'Adduct', 'Relative_Error']].to_string(index=False))
        high_error_ions[['smiles', 'Adduct', 'Relative_Error']].to_csv('high_error_ions.csv', index=False)
    else:
        print("No ions with relative error > 15% found.")

    # First figure: Positive vs Negative ions
    plt.figure(figsize=(3, 3))
    sns.violinplot(
        x="Polarity", 
        y="Relative_Error",
        data=df,
        palette="Set2",
        cut=0,
        width=0.8,
        inner="quartile"
    )
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    # plt.title("Positive vs Negative Ions", pad=20, fontproperties=arial_font, fontsize=22)
    # plt.xlabel("Ion Polarity", fontproperties=arial_font, fontsize=20)
    # plt.ylabel("Relative Error (%)", fontproperties=arial_font, fontsize=20)

    plt.ylim(-16, 16)
    plt.tick_params(axis='x', rotation=45)

    for label in plt.gca().get_yticklabels():
        label.set_fontproperties(arial_font)
        label.set_fontsize(18)

    # Remove right and top borders
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # remove x lable
    plt.gca().set_xlabel('')  # 移除x轴标签
    plt.gca().set_ylabel('')  # 移除x轴标签
    plt.gca().set_xticklabels([]) 

    plt.tight_layout()
    plt.savefig(f'./positive_vs_negative{fig_name}.png', dpi=600, bbox_inches='tight')

    # Second figure: Different Adduct types
    plt.figure(figsize=(4, 3))
    sns.violinplot(
        x="Adduct", 
        y="Relative_Error",
        data=df,
        palette="Set3",
        cut=0,
        inner="quartile",
        width = 0.8
    )

    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.yticks([-15,-10, -5, 0, 5, 10,15])  # 仅显示这5个刻度
    plt.tick_params(axis='x', rotation=45)

    for label in plt.gca().get_yticklabels():
        label.set_fontproperties(arial_font)
        label.set_fontsize(18)

    # Remove right and top borders
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_xlabel('')  # 移除x轴标签
    plt.gca().set_ylabel('')  # 移除x轴标签
    plt.gca().set_xticklabels([]) 

    plt.ylim(-16, 16)
    plt.tight_layout()
    plt.savefig(f'./adduct_types{fig_name}.png', dpi=600, bbox_inches='tight')


def plot_in_house_compare(data_path,fig_name ):

    data = pd.read_csv(data_path)
    # 计算相对误差
    data['HyperCCS_RE'] = np.abs(data['hyperccs'] - data['ccs']) / data['ccs'] * 100
    data['ALLCCS2_RE'] = np.abs(data['allccs2'] - data['ccs']) / data['ccs'] * 100

    # 设置图表样式
    plt.style.use('default')  # 重置为默认样式
    plt.figure(figsize=(10, 6))

    # 绘制折线图（取消网格和边框）
    ax = plt.gca()
    ax.spines['right'].set_visible(False)  # 移除右边框
    ax.spines['top'].set_visible(False)    # 移除上边框
    ax.grid(False)                         # 关闭背景虚线

    # 生成颜色列表，每个代谢物一个颜色
    num_metabolites = len(data)
    colors = plt.cm.tab20(np.linspace(0, 1, num_metabolites))

    # 设置横轴位置（更靠近中间的位置）
    x_positions = [0.95, 1.05]  # 调整后的位置，更靠近中心
    x_labels = ['ALLCCS2', 'HyperCCS']  # 交换顺序

    # 为每个代谢物绘制连接线
    for i, (idx, row) in enumerate(data.iterrows()):
        y_values = [row['ALLCCS2_RE'], row['HyperCCS_RE']]  # 交换顺序
        plt.plot(x_positions, y_values, 
                marker='o', markersize=8,
                linestyle='-', color=colors[i],
                linewidth=1.5, label=row['Name'])

    # # 设置横轴范围，使两个标签更集中
    plt.xlim(0.8, 1.2)

    # 设置横轴标签和标题
    plt.xticks(x_positions, x_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Relative Error (%)', fontsize=20)

    # 调整图例位置和样式
    plt.legend(frameon=False, fontsize=13, loc='upper left')

    # 调整布局防止标签被裁剪
    plt.tight_layout()

    # 保存图像（分辨率 300 DPI，透明背景可选）
    plt.savefig(f"{fig_name}.png", dpi=600, bbox_inches='tight', transparent=False)
    
    
def plot_relative_error_boxplot(datasets, dataset_labels, colors, 
                              title='relative_error_boxplot', xlabel='Models', ylabel='Relative_error (%)', 
                              figsize=(8, 8), 
                              fontsize_title=26, 
                              fontsize_label=26,
                              name='METLIN'):
    """
    绘制箱线图并计算各模型的IQR统计量
    
    Parameters:
    -----------
    datasets : list of str
        包含CCS预测结果的CSV文件路径列表
    dataset_labels : list of str
        每个模型/数据集的标签列表
    colors : list of str
        每个箱线图的颜色列表（十六进制格式）
    title : str, optional
        图表标题
    figsize : tuple, optional
        图表尺寸（宽, 高）
    name : str, optional
        输出图片名称（不含扩展名）

    Returns:
    --------
    pd.DataFrame
        包含各模型Q1、Q3和IQR的统计量表
    """
    
    # 数据读取与处理
    all_data = []
    iqr_stats = []
    
    for dataset, label in zip(datasets, dataset_labels):
        df = pd.read_csv(dataset)
        df['relative_error'] = abs(df['true_ccs'] - df['predicted_ccs']) / df['true_ccs'] * 100
        df['dataset'] = label
        
        # 计算IQR统计量
        q1 = df['relative_error'].quantile(0.25)
        q3 = df['relative_error'].quantile(0.75)
        iqr_stats.append({
            'Model': label,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            'IQR': round(q3 - q1, 2)
        })
        all_data.append(df)
    
    # 合并数据并创建统计表
    iqr_df = pd.DataFrame(iqr_stats)
    all_data = pd.concat(all_data)

    # 绘制箱线图
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        x='dataset', 
        y='relative_error', 
        data=all_data, 
        palette=dict(zip(dataset_labels, colors)),
        showfliers=False,
        boxprops=dict(edgecolor='none'))
    
    # 图形美化
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # 设置字体样式
    plt.title(title, fontname='Arial', fontsize=fontsize_title, pad=20)
    plt.xlabel(xlabel, fontname='Arial', fontsize=fontsize_label, labelpad=20)
    plt.ylabel(ylabel, fontname='Arial', fontsize=fontsize_label, labelpad=20)
    
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
    
    # 保存输出
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', labelsize=30)
    plt.tight_layout()
    plt.savefig(f"./results/{name}.png", dpi=500)
    plt.close()
    
    return iqr_df


def plot_mz_ccs_compare(data_path,fig_name):

    # 设置边框可见性
    plt.figure(figsize=(8, 8))

    data = pd.read_csv(data_path)
    # 提取m/z和CCS_AVG列
    mz = data['m/z']
    ccs = data['CCS']

    # 线性回归拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(mz, ccs)
    line = slope * mz + intercept
    fit_label = f'Fit line: y = {slope:.4f}x + {intercept:.2f}\nR² = {r_value**2:.4f}'
    ax = plt.gca()
    ax.plot(mz, line, 'r-')
    

    # 绘制散点图
    scatter_color = '#688ce6'
    plt.scatter(mz, ccs, color=scatter_color, label='Data points',s=12, alpha=0.6)

    # 绘制拟合直线
    custom_legend = [
        Line2D([0], [0], color='#688ce6', lw=0, marker='o', markersize=6, label='Data points'),
        Line2D([0], [0], color='red', lw=2, label=fit_label)
    ]
    ax.legend(handles=custom_legend, fontsize=14,loc='upper left')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='both', labelsize=24)  # 统一设置刻度字号
    # 显示图形
    plt.savefig(f"./results/{fig_name}.png", dpi=500)


def plot_adduct_re(df):
    # 设置图形大小
    plt.figure(figsize=(4, 7))
    
    # 获取唯一的数据集名称和离子类型
    datasets = df['Dataset'].unique()
    ions = df['Ion'].unique()
    
    # 设置颜色
    colors = {'ALLCCS2': '#5391f5', 'METLIN': '#f1ca5c'}
    
    # 为每个离子类型创建位置
    y_pos = np.arange(len(ions))
    
    # 绘制每个数据集的柱状图
    for i, ion in enumerate(ions):
        for dataset in datasets:
            # 获取当前离子和数据集的数据
            data = df[(df['Ion'] == ion) & (df['Dataset'] == dataset)]
            if not data.empty:
                error = data['Median Error (%)'].values[0]
                # 绘制柱状图
                plt.barh(
                    y=i, 
                    width=error, 
                    height=0.6,
                    color=colors[dataset], 
                    alpha=0.6, 
                    label=dataset if i == 0 else ""
                )
    
    # 设置y轴刻度和标签
    # plt.yticks(y_pos, ions)
    plt.yticks(y_pos) 
    ax = plt.gca()
    ax.set_xticklabels([])  # 横坐标
    ax.set_yticklabels([])  # 纵坐标
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[d], alpha=0.7) for d in datasets]
    plt.legend(handles, datasets, loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig("./results/adduct_re.png", dpi=500)


def plot_residuals(data_path, fig_name):
    """
    绘制残差图并保存为图片
    
    参数:
    data_path -- CSV文件路径，需包含true_ccs和predicted_ccs列
    fig_name -- 输出图片文件名（可包含路径）
    """
    # 读取数据
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    
    # 检查必要列是否存在
    required_cols = ['true_ccs', 'predicted_ccs', 'adducts']
    if not all(col in data.columns for col in required_cols):
        print(f"CSV文件必须包含以下列: {required_cols}")
        return
    
    # 计算残差
    data['residual'] = data['true_ccs'] - data['predicted_ccs']
    
    # 创建画布
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # 定义颜色方案
    # colors = ['#f1ca5c', '#7e9abf', '#82bd86', '#aa2195', '#5391f5']
    colors= [
    '#f1ca5c',  # 柔和亮黄
    '#d8573c',  # 赤陶橙红（暖色对比）
    '#aa2195',  # 紫红色
    '#c4b7d4',  # 薰衣草灰（紫色延展）
    '#7e9abf',  # 雾霾蓝
    '#5391f5',  # 中亮蓝
    '#3c3f58',  # 深靛蓝灰
    '#415753',  # 深青灰
    '#8aaf9c',  # 柔和灰绿
    '#82bd86'   # 柔和绿色
    ]
    adducts = data['adducts'].unique()
    adduct_colors = {adduct: colors[i % len(colors)] for i, adduct in enumerate(adducts)}
    
    # 绘制散点图（按离子类型着色）
    for adduct in adducts:
        subset = data[data['adducts'] == adduct]
        ax.scatter(subset['m/z'], subset['residual'], 
                  c=adduct_colors[adduct], label=adduct, s=20, alpha=0.5)
    
    # # 添加平滑曲线（改为虚线）
    # sns.regplot(
    #     x='predicted_ccs', 
    #     y='residual', 
    #     data=data,
    #     lowess=True,
    #     scatter=False,
    #     line_kws={'color': 'red', 'lw': 1, 'linestyle': '--'}  # 改为虚线
    # )
    
    # 添加y=0的参考线（实线）
    ax.axhline(
        y=0, 
        color='gray', 
        linestyle='-',  # 实线
        linewidth=1,
        alpha=0.7
    )

    ax.set_yticks([-30, -20, -10, 0, 10, 20, 30])  # 设置刻度位置
    ax.set_ylim(-30, 30)  # 设置y轴范围
    
    # 添加图表元素
    plt.xlabel('m/z')
    plt.ylabel('Residual (True - Predicted)')
    # plt.legend()
    
    # 美化图表
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig(fig_name, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"残差图已保存至: {fig_name}")