import pandas as pd

sites = {1: "Google", 2: "Baidu", 3: "Wiki"}

myvar1 = pd.Series(sites, index=[1, 2, 3], name="best")
print(myvar1)

myvar2 = pd.Series(sites, index=[1, 2], name="good")
print(myvar2)
