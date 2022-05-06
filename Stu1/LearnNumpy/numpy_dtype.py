import numpy as np

# 创建数组对象 ndarray
a = np.array([1, 2, 3])
print(a)
print('数据类型:', a.dtype)

# 修改数据类型为 float32
a = a.astype(np.float32)
print(a)
print('新的数据类型:', a.dtype)

# 创建数据类型为 int16 的数组
b = np.array([1, 2, 3], dtype=np.int16)
print('b:', b)
print('b的数据类型:', b.dtype)
