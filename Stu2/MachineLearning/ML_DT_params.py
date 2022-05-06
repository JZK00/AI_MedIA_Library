from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
决策树模型
参数调整
"""

# 训练集路径
train_path = r'../../datasets/breast_cancer/feature_screening/LASSO/train_screen.xlsx'

# 测试集路径
test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'

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

# 调参
scoreTrainList, scoreTestList = [], []
for i in range(1, 10):
    clf_dt = DecisionTreeClassifier(criterion='entropy'
                                    , random_state=11
                                    , max_depth=i
                                    , min_samples_leaf=5,
                                    min_samples_split=25
                                    )
    clf_dt.fit(X_train, y_train)
    score_train = clf_dt.score(X_train, y_train)
    score_test = clf_dt.score(X_test, y_test)  # 准确率
    scoreTrainList.append(score_train)
    scoreTestList.append(score_test)
    print(score_train, score_test)

plt.plot(range(1, 10), scoreTrainList, label='Train')
plt.plot(range(1, 10), scoreTestList, label='Test')
plt.legend()
plt.show()
