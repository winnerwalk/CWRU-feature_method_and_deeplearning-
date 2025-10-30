import pandas as pd

# 加载数据
data = pd.read_csv('./data_with_feature/data_with_feature.csv')

print('数据基本信息：')
data.info()

# 查看数据集行数和列数
rows, columns = data.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(data.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(data.head().to_csv(sep='\t', na_rep='nan'))

from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# 提取特征和目标变量
X = data.drop(['data', 'label'], axis=1)
y = data['label']

# 计算各特征与 label 的相关性系数
correlation_with_label = data.drop(columns=['data']).corr()['label'].drop('label')

# 获取相关性绝对值最大的前 20 个特征
top_10_features = correlation_with_label.abs().nlargest(20).index

print('筛选出的 10 个特征：', top_10_features)