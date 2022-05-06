import pandas as pd

"""
变量删除

二八法则

删除标准（仅供参考）：
- 变量的缺失值数量超过总样本量的80%。
- 离散变量中某个类别的数量超过该变量总数的80%。

"""


def missing_sample():
    """
    变量的缺失值数量超过总样本量的80%
    """
    threshold = 0.8  # 阈值

    # 文件路径
    file_path = r"../../datasets/missing_delete.xlsx"

    # 填充后的文件保存路径
    save_path = r"../../datasets/missing_delete_sample.xlsx"

    # 读取数据
    df = pd.read_excel(file_path)

    missing_num = df.isnull().sum(axis=0)  # 计算每个变量的缺失值数量
    print("缺失值数量:\n", missing_num)

    missing_ratio = missing_num / df.shape[0]  # 缺失比例
    print("缺失比例:\n", missing_ratio)

    missing_col = missing_ratio[missing_ratio > threshold].index.tolist()  # 大于阈值的列
    print("大于阈值的列:", missing_col)

    df.drop(columns=missing_col, inplace=True)  # 删除大于阈值的列

    df.to_excel(save_path, index=False)
    print('finish.')


def single_focus():
    """
    离散变量中某个类别的数量超过该变量总数的80%
    """
    threshold = 0.8  # 阈值

    # 文件路径
    file_path = r"../../datasets/missing_delete.xlsx"

    # 填充后的文件保存路径
    save_path = r"../../datasets/missing_delete_focus.xlsx"

    # 读取数据
    df = pd.read_excel(file_path)

    del_col = []  # 存储需要删除的列
    for col in df.columns:
        max_num = df[col].value_counts()  # 单个类别每一类的数量
        num_ratio = max_num.max() / max_num.sum()  # 最多类别占总数的比例
        print(f"{col}: 单个类别数最高占比为 {num_ratio}")

        if num_ratio > threshold:
            del_col.append(col)

    # 删除满足条件的列
    df.drop(columns=del_col, inplace=True)
    print("删除满足条件的列:", del_col)

    df.to_excel(save_path, index=False)
    print('finish.')


if __name__ == '__main__':
    missing_sample()
    single_focus()
