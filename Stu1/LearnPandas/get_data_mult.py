import pandas as pd

data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data, index=['1st', '2nd', '3rd'])
print(df)

print('返回第一行和第二行:')
print(df.iloc[[0, 1]])
print(df.loc[['1st', '2nd']])  # 指定行名

print('返回第一列:')
print(df.iloc[:, 0])
print(df.loc[:, 'calories'])  # 指定列名

print('返回第2行，第1列')
print(df.iloc[1, 0])
print(df.loc['2nd', 'calories'])
