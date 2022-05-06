import pandas as pd
from scipy import stats

"""
相关性分析
"""
# 文件路径
file_path = r'../../datasets/breast_cancer/breast_cancer.xlsx'

# 计算结果保存路径
save_result = r'../../datasets/breast_cancer/breast_cancer_corr.xlsx'

# 读取数据
df = pd.read_excel(file_path)
print(df)

# 取出特征
feature = df.iloc[:, :-1]  # 最后一列是标签，不取
print('data:\n', feature)

# 相关性分析
method = 'spearman'  # method : {'pearson', 'spearman', 'kendall'}  # 选择方法

coor = []
for i in feature.columns:
    for j in feature.columns:
        df_1 = feature[i].values
        df_2 = feature[j].values

        # 选择方法进行计算
        if method == 'pearson':
            r, p = stats.pearsonr(df_1, df_2)
        elif method == 'spearman':
            r, p = stats.spearmanr(df_1, df_2, nan_policy='omit')
        elif method == 'kendall':
            r, p = stats.kendalltau(df_1, df_2, nan_policy='omit')
        else:
            print("method方法选择错误")

        coor.append([i, j, r, p])

# 保存计算结果
df_corr = pd.DataFrame(coor, columns=["Var_1", "Var_2", "R", "P"])
df_corr.to_excel(save_result, index=False)
print('finish.')
