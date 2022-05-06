import cv2
import numpy as np
import torch
import torch.nn as nn

"""

卷积神经网络  Convolutional Neural Network
"""

img = cv2.imread('lenna.jpg')
print(img)
print("原图形状:", img.shape) # H W C

cv2.imshow('img', img)
# cv2.waitKey()  # 等待

# H W C -> N C H W   ( 1 3 622 580)
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img)
img = img.unsqueeze(0)  # 加轴

# *******************************************
# 卷积 池化层
# cnn = nn.Conv2d(3, 3, 3, 1)  # 卷积
# cnn = nn.MaxPool2d(2, 2)  # 最大池化
cnn = nn.AvgPool2d(2, 2)  # 平均池化

# *******************************************

img = cnn(img / 255.)

# N C H W -> H W C
img = img.squeeze(0)
img = img.detach().numpy()
img = np.transpose(img, (1, 2, 0))
img = img * 255.
img = img.astype(np.uint8)
print("新图形状:", img.shape)

cv2.imshow('new', img)
cv2.waitKey()
