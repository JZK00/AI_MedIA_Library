import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""
数据压缩
"""

# 训练集路径
train_path = r'../../datasets/breast_cancer/split_datasets/train.xlsx'

# 测试集路径
test_path = r'../../datasets/breast_cancer/split_datasets/test.xlsx'

# 结果保存目录
save_dir = r'../../datasets/breast_cancer/split_datasets'
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

# ============ 数据压缩  只压缩特征，不压缩标签 ==========
# scaler = MinMaxScaler()  # 归一化压缩
scaler = StandardScaler()  # 标准化压缩

scaler.fit(X_train)  # 只能为训练集，不能为测试集

X_train_comp = scaler.transform(X_train)  # 压缩训练集特征
df_train_comp = pd.concat([pd.DataFrame(X_train_comp, columns=X_train.columns), y_train], axis=1)
print('训练集压缩:\n', df_train_comp)

X_test_comp = scaler.transform(X_test)  # 压缩测试集特征
df_test_comp = pd.concat([pd.DataFrame(X_test_comp, columns=X_test.columns), y_test], axis=1)
print('测试集压缩:\n', df_test_comp)

# 保存结果
df_train_comp.to_excel(os.path.join(save_dir, "train_comp.xlsx"), index=False)
df_test_comp.to_excel(os.path.join(save_dir, "test_comp.xlsx"), index=False)
print('save successfully.')
