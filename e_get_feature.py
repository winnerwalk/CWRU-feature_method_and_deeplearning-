import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def extract_features(data):
    """提取时序和频域特征"""
    features = []
    
    # 转换为numpy数组
    x = np.array(data)
    
    # 1. 时序特征
    # 基本统计特征
    features.append(np.max(x))  # 最大值
    features.append(np.min(x))  # 最小值
    features.append(np.mean(x))  # 均值
    features.append(np.sqrt(np.mean(x**2)))  # 均方根
    features.append(np.std(x))  # 标准差
    
    # 高阶统计特征
    skewness = stats.skew(x) if len(x) > 2 else 0
    kurtosis = stats.kurtosis(x) if len(x) > 3 else 0
    features.append(skewness)  # 偏度
    features.append(kurtosis)  # 峰度
    
    # 形状因子
    rms = np.sqrt(np.mean(x**2))
    mean_abs = np.mean(np.abs(x))
    features.append(kurtosis + 3 if len(x) > 3 else 0)  # 峰度因子
    features.append(rms / mean_abs if mean_abs != 0 else 0)  # 形状因子
    features.append(np.max(np.abs(x)) / mean_abs if mean_abs != 0 else 0)  # 脉冲因子
    
    # 裕度因子
    if np.mean(np.sqrt(np.abs(x))) != 0:
        clearance_factor = np.max(np.abs(x)) / (np.mean(np.sqrt(np.abs(x)))**2)
    else:
        clearance_factor = 0
    features.append(clearance_factor)  # 裕度因子
    
    # 其他时序特征
    features.append(np.max(np.abs(x)))  # 绝对最大值
    features.append(np.sum(np.abs(np.diff(x))))  # 波动变化和
    
    # 线性趋势
    if len(x) > 1:
        t = np.arange(len(x))
        slope, intercept, _, _, _ = stats.linregress(t, x)
        features.append(slope)  # 线性趋势
    else:
        features.append(0)
    
    # 自相关系数 (滞后1)
    if len(x) > 1:
        autocorr = np.corrcoef(x[:-1], x[1:])[0,1] if not np.isnan(np.corrcoef(x[:-1], x[1:])[0,1]) else 0
        features.append(autocorr)
    else:
        features.append(0)
    
    # 序列非线性 (用偏度的绝对值)
    features.append(np.abs(skewness))
    
    # 时间复杂度估计 (用标准差与均方根的比值)
    complexity = np.std(x) / rms if rms != 0 else 0
    features.append(complexity)
    
    # 高于均值的数目
    features.append(np.sum(x > np.mean(x)))
    
    # 平均超过一阶差异
    features.append(np.mean(np.abs(np.diff(x))) if len(x) > 1 else 0)
    
    # 峰值数
    if len(x) > 1:
        peaks, _ = find_peaks(np.abs(x), height=np.std(x)*0.5)
        features.append(len(peaks))
    else:
        features.append(0)
    
    # 2. 频域特征
    # 傅里叶变换
    if len(x) > 0:
        fft_vals = np.abs(fft(x))
        # 取一半（对称性）
        fft_vals = fft_vals[:len(fft_vals)//2]
        
        if len(fft_vals) > 0:
            features.append(np.max(fft_vals))  # 频域最大值
            features.append(np.min(fft_vals))  # 频域最小值
            features.append(np.mean(fft_vals))  # 频域均值
            features.append(np.sqrt(np.mean(fft_vals**2)))  # 频域均方根
            features.append(np.std(fft_vals))  # 频域标准差
            
            # 频域偏度和峰度
            if len(fft_vals) > 2:
                features.append(stats.skew(fft_vals))
            else:
                features.append(0)
                
            if len(fft_vals) > 3:
                features.append(stats.kurtosis(fft_vals))
            else:
                features.append(0)
        else:
            features.extend([0] * 7)
    else:
        features.extend([0] * 7)
    
    return features

# 读取dataset.csv文件
try:
    # 尝试不同的编码方式读取CSV文件
    df = pd.read_csv('.\dataset\dataset_.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('.\dataset\dataset_.csv', encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv('.\dataset\dataset_.csv', encoding='latin-1')

print("CSV文件读取成功！")
print(f"数据形状: {df.shape}")

# 查看列名，确定数据列和标签列
print("列名:", df.columns.tolist())

# 假设数据列名为'data'，标签列名为'label'
# 如果列名不同，请根据实际情况修改
data_column = 'data'
label_column = 'label'

# 检查列是否存在
if data_column not in df.columns:
    # 尝试找到包含数据的列
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['data', 'signal', 'value']):
            data_column = col
            break

if label_column not in df.columns:
    # 尝试找到标签列
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['label', 'class', 'target']):
            label_column = col
            break

print(f"使用数据列: {data_column}")
print(f"使用标签列: {label_column}")

# 提取数据和标签
data_list = []
labels = []

for idx, row in df.iterrows():
    # 提取数据（假设数据是字符串形式的列表）
    data_str = str(row[data_column])
    
    # 清理数据字符串并转换为列表
    # 移除引号和括号
    data_str = data_str.replace('"[', '').replace(']"', '').replace('[', '').replace(']', '')
    
    # 分割字符串并转换为浮点数
    try:
        data = [float(x) for x in data_str.split()]
        data_list.append(data)
        labels.append(row[label_column])
    except ValueError as e:
        print(f"警告: 第 {idx} 行数据格式错误: {e}")
        continue

print(f"成功解析 {len(data_list)} 个数据样本")

# 定义特征名称
time_features = [
    'max', 'min', 'mean', 'rms', 'std', 'skewness', 'kurtosis', 
    'kurtosis_factor', 'shape_factor', 'impulse_factor', 'clearance_factor',
    'abs_max', 'fluctuation_sum', 'linear_trend', 'autocorrelation',
    'nonlinearity', 'complexity', 'above_mean_count', 'mean_diff', 'peak_count'
]

freq_features = [
    'freq_max', 'freq_min', 'freq_mean', 'freq_rms', 'freq_std',
    'freq_skewness', 'freq_kurtosis'
]

all_features = time_features + freq_features

# 提取所有特征
feature_data = []
for i, data in enumerate(data_list):
    if i % 100 == 0:
        print(f"正在处理第 {i+1}/{len(data_list)} 个样本...")
    features = extract_features(data)
    feature_data.append(features)

# 创建DataFrame
df_results = pd.DataFrame(feature_data, columns=all_features)

# 添加原始数据和标签
df_results['data'] = data_list
df_results['label'] = labels

# 重新排列列的顺序，让数据和标签在前，特征在后
columns_order = ['data', 'label'] + all_features
df_results = df_results[columns_order]

# 保存到CSV文件
df_results.to_csv('./data_with_feature/data_with_feature.csv', index=False)
print("特征提取完成！结果已保存到 data_with_feature.csv")

# 显示特征统计信息
print(f"\n特征维度: {df_results.shape}")
print(f"时序特征数量: {len(time_features)}")
print(f"频域特征数量: {len(freq_features)}")
print(f"总特征数量: {len(all_features)}")

# 显示前几行数据
print("\n前5个样本的特征预览:")
print(df_results.head())