import pandas as pd

data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)
print(df)

# 返回第一行
print(df.loc[0])

# 返回第二行
print(df.loc[1])
