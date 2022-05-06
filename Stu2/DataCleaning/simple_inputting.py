import pandas as pd

"""
简单填充

数据填充时，可以采用：

- 连续变量，使用均值填充。
- 无序变量，使用众数填充。
- 有序变量，使用中位数填充。

"""

# 文件路径
file_path = r"../../datasets/missing_value.xlsx"

# 填充后的文件保存路径
save_path = r"../../datasets/missing_value_inputting.xlsx"

# 连续变量，如果没有，则 continuous_var = []
continuous_var = ['X0', 'X1']

# 无序变量，如果没有，则 unordered_var = []
unordered_var = ['X2', 'X3']

# 有序变量 如果没有，则 ordinal_var = []
ordinal_var = ['X4']

# 读取数据
df = pd.read_excel(file_path)
print(df)

for col in df.columns:

    if df[col].isnull().sum() == 0:
        print(f"{col}: 无缺失值")
        continue

    # 连续变量，使用均值填充
    if col in continuous_var:
        mean = df[col].mean()
        df[col] = df[col].fillna(value=mean)
        print(f"{col}: 均值填充，填充值为: {mean}")

    # 无序变量，使用众数填充
    elif col in unordered_var:
        mode = df[col].mode().values[0]
        df[col] = df[col].fillna(value=mode)
        print(f"{col}: 众数填充，填充值为: {mode}")

    # 有序变量，使用中位数填充
    elif col in ordinal_var:
        median = df[col].median()
        df[col] = df[col].fillna(value=median)
        print(f"{col}: 中位数填充，填充值为: {median}")

    else:
        print(f"{col}: 不明确变量类型")

# 保存填充后的值
df.to_excel(save_path, index=False)
