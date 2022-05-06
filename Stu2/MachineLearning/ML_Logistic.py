import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 训练集路径
train_path = r'../../datasets/breast_cancer/feature_screening/LASSO/train_screen.xlsx'

# 测试集路径
test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'

# 结果保存目录
save_dir = r'../../datasets/breast_cancer/machine_learning/Logistic'
os.makedirs(save_dir, exist_ok=True)

# 训练集
df_train = pd.read_excel(train_path)  # pd.read_csv()
X_train = df_train.iloc[:, :-1]  # 训练集特征
y_train = df_train.iloc[:, -1]  # 训练集标签
print('train datasets:\n', df_train)

# 测试集
df_test = pd.read_excel(test_path)
X_test = df_test.iloc[:, :-1]  # 测试集特征
y_test = df_test.iloc[:, -1]  # 测试集标签
print('test datasets:\n', df_test)

# 机器学习建模
estimator = LogisticRegression(max_iter=1000)
estimator.fit(X_train, y_train)

# 预测
y_pred_test = estimator.predict(X_test).tolist()  # 预测结果
print('y_test:', y_test.values.tolist())
print('y_pred_test:', y_pred_test)

y_score_test = estimator.predict_proba(X_test)[:, 1].tolist()  # 预测为1的概率
print('y_score_test:', y_score_test)


# ====================== 模型评估 ======================
# 混淆矩阵
conf_m = metrics.confusion_matrix(y_test, y_pred_test)
TP = conf_m[1][1]  # 真阳性
FP = conf_m[0][1]  # 假阳性
TN = conf_m[0][0]  # 真阴性
FN = conf_m[1][0]  # 假阴性

plt.clf()
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)  # 横向真实值，纵向预测值
plt.savefig(os.path.join(save_dir, "confusion_matrix.jpg"), dpi=300)

# ROC曲线
plt.clf()
metrics.RocCurveDisplay.from_predictions(y_test, y_score_test)
plt.savefig(os.path.join(save_dir, "roc.jpg"), dpi=300)

# 评估报告
report = metrics.classification_report(y_test, y_pred_test, digits=4)  # 评估报告
Sensitivity = TP / (TP + FN)  # 敏感度 召回率
Specificity = TN / (TN + FP)  # 特异度
AUC = metrics.roc_auc_score(y_test, y_score_test)

with open(os.path.join(save_dir, "report.txt"), mode='w', encoding='utf-8') as fw:
    fw.write(report)
    fw.write('\n')
    fw.write(f"Sensitivity: {Sensitivity}\n")
    fw.write(f"Specificity: {Specificity}\n")
    fw.write(f"AUC: {AUC}\n")

print('report:\n', report)
print('Sensitivity:', Sensitivity)
print('Specificity:', Specificity)
print('AUC:', AUC)
