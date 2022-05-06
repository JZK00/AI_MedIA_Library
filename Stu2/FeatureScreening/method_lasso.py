from sklearn.linear_model import LassoCV
import pandas as pd
import os

# 训练集路径
train_path = r'../../datasets/breast_cancer/split_datasets/train_comp.xlsx'

# 测试集路径
test_path = r'../../datasets/breast_cancer/split_datasets/test_comp.xlsx'

# 结果保存目录
save_dir = r'../../datasets/breast_cancer/feature_screening/LASSO'
os.makedirs(save_dir, exist_ok=True)

# 训练集
df_train = pd.read_excel(train_path)
X_train = df_train.iloc[:, :-1]  # 训练集特征
y_train = df_train.iloc[:, -1]  # 训练集标签
print('train datasets:\n', df_train)

# 测试集
df_test = pd.read_excel(test_path)
X_test = df_test.iloc[:, :-1]  # 测试集特征
y_test = df_test.iloc[:, -1]  # 测试集标签
print('test datasets:\n', df_test)

# 特征筛选
estimator = LassoCV(cv=5, max_iter=1e6)  # cv: 交叉验证次数
estimator.fit(X_train, y_train)

coef = estimator.coef_  # 系数不为 0 的特征是特征选择后的结果
print('coef:', coef)

screen_features = X_train.columns[coef != 0]
print('screen_features:', screen_features)

df_train_screen = pd.concat([X_train.loc[:, screen_features], y_train], axis=1)
print('训练集筛选:\n', df_train_screen)

df_test_screen = pd.concat([X_test.loc[:, screen_features], y_test], axis=1)
print('测试集筛选:\n', df_test_screen)

# 保存结果
df_train_screen.to_excel(os.path.join(save_dir, "train_screen.xlsx"), index=False)
df_test_screen.to_excel(os.path.join(save_dir, "test_screen.xlsx"), index=False)
print('save successfully.')
