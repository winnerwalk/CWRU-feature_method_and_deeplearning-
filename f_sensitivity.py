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

# 创建 SVM 分类器
svc = SVC(kernel="linear")

# 创建 RFE 对象并指定要选择的特征数量
n_features_to_select = 20
rfe = RFE(estimator=svc, n_features_to_select=n_features_to_select)

# 执行特征选择
rfe.fit(X, y)

# 输出选择的特征和特征排名
selected_features = X.columns[rfe.support_]
feature_ranking = rfe.ranking_

print(f"选择的特征: {selected_features}")
print(f"特征排名: {feature_ranking}")