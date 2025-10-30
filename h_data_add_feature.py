import pandas as pd

# 读取文件
df = pd.read_csv('./data_with_feature/data_with_feature.csv')

# 要添加的列名
columns_to_add = ['max', 'min', 'mean', 'skewness', 'kurtosis', 'kurtosis_factor',
                  'shape_factor', 'impulse_factor', 'clearance_factor', 'abs_max',
                  'autocorrelation', 'nonlinearity', 'complexity', 'freq_max',
                  'freq_min', 'freq_mean', 'freq_rms', 'freq_std', 'freq_skewness',
                  'freq_kurtosis', 'peak_count', 'autocorrelation', 'freq_kurtosis',
                  'freq_skewness', 'min', 'abs_max', 'max', 'complexity',
                  'shape_factor', 'freq_max', 'nonlinearity', 'freq_std', 'freq_rms',
                  'rms', 'std', 'clearance_factor', 'freq_mean', 'kurtosis_factor',
                  'kurtosis', 'impulse_factor']

# 定义函数，将列值添加到data列的列表中
def append_values(row):
    data_list = eval(row['data'])
    for col in columns_to_add:
        data_list.append(row[col])
    return str(data_list)

# 应用函数到每一行
df['data'] = df.apply(append_values, axis=1)

# 选择需要的列
result_df = df[['data', 'label']]

# 将结果保存为csv文件
csv_path = './data_with_feature/data_add_feature.csv'
result_df.to_csv(csv_path, index=False)