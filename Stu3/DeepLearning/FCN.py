import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""
全连接神经网络  Fully connected neural network
"""

x = torch.arange(100) / 200.
y = (3 * x + torch.randn(100) / 10.) / 10.

print(x)
print(y)

plt.scatter(x, y)
plt.show()


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.layers = nn.Sequential(

            # 三步走
            # 全连接 -> 标准化 -> 激活函数

            nn.Linear(1, 16),
            nn.BatchNorm1d(16),  # 无标准化的效果
            nn.ReLU(),  # 无激活函数的效果

            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),


            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


net = MainNet()
loss_fn = nn.MSELoss()  # 损失函数  均方差  (y' - y)^2

# nn.L1Loss()
# nn.CrossEntropyLoss()
# nn.BCELoss()

optimizer = optim.Adam(net.parameters())  # 优化器

losses = []
for epoch in range(1000):
    print('epoch:', epoch)

    # 前向传播
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    out = net(x)  # N V

    # 求损失
    loss = loss_fn(out, y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('loss:', loss)

    # 可视化
    plt.clf()
    plt.scatter(x, y)
    plt.scatter(x, out.detach().numpy())
    plt.pause(0.1)

    # 画损失
    # losses.append(loss.item())
    # plt.clf()  # 清空
    # plt.plot(losses)
    # plt.pause(0.1)
