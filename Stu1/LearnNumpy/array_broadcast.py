import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a * b
print(c)

print("=====当运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制。=====")
a = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
b = np.array([0, 1, 2])
print(a)
print(b)
print(a + b)
