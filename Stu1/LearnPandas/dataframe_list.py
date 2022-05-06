import pandas as pd

data = [['Google', 10], ['Baidu', 12], ['Wiki', 13]]

df = pd.DataFrame(data, columns=['Site', 'Age'])

print(df)
