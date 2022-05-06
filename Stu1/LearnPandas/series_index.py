import pandas as pd

a = ["Google", "Baidu", "Wiki"]

myvar = pd.Series(a, index=["x", "y", "z"])

print(myvar)
print(myvar["y"])  # 指定索引名称
print(myvar[1])  # 指定索引
