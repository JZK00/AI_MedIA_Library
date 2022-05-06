import os
import pandas as pd
from sklearn.model_selection import train_test_split

"""
拆分数据集
"""
# 文件路径
file_path = r'../../datasets/breast_cancer/breast_cancer.xlsx'

# 计算结果保存目录
save_dir = r'../../datasets/breast_cancer/split_datasets'
os.makedirs(save_dir, exist_ok=True)

# 拆分比例  测试集样本量
# 如果为0.2, 说明测试集占总样本量的20%, 训练集占总样本量的80%
ratio = 0.2

df = pd.read_excel(file_path)
X = df.iloc[:, :-1]  # 特征
y = df.iloc[:, -1]  # 标签
print('feature:\n', X)
print('label:\n', y)

# 拆分数据集
# 如果不是分类预测， stratify=None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42,
                                                    stratify=y)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
print('train:\n', df_train)
print('test:\n', df_test)

# 保存结果
df_train.to_excel(os.path.join(save_dir, 'train.xlsx'), index=False)
df_test.to_excel(os.path.join(save_dir, 'test.xlsx'), index=False)
print('save successfully.')
