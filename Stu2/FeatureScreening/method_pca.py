from sklearn.decomposition import PCA
import pandas as pd
import os

# 训练集路径
train_path = r'../../datasets/breast_cancer/split_datasets/train_comp.xlsx'

# 测试集路径
test_path = r'../../datasets/breast_cancer/split_datasets/test_comp.xlsx'

# 结果保存目录
save_dir = r'../../datasets/breast_cancer/feature_screening/PCA'
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
estimator = PCA(n_components=25)  # n_components 指定希望PCA降维后的特征维度数目

estimator.fit(X_train)

X_train_screen = estimator.transform(X_train)  # 筛选训练集特征
df_train_screen = pd.concat([pd.DataFrame(X_train_screen, columns=[f"X{i}" for i in range(X_train_screen.shape[1])]),
                             y_train], axis=1)
print('训练集筛选:\n', df_train_screen)

X_test_screen = estimator.transform(X_test)  # 筛选测试集特征
df_test_screen = pd.concat([pd.DataFrame(X_test_screen, columns=[f"X{i}" for i in range(X_test_screen.shape[1])]),
                            y_test], axis=1)
print('测试集筛选:\n', df_test_screen)

# 保存结果
df_train_screen.to_excel(os.path.join(save_dir, "train_screen.xlsx"), index=False)
df_test_screen.to_excel(os.path.join(save_dir, "test_screen.xlsx"), index=False)
print('save successfully.')

# 降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
print(estimator.explained_variance_)

# 降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。
print(estimator.explained_variance_ratio_)
