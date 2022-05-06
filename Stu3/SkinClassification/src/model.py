#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# 轻量级网络，速度很快，适合于移动端
from torchvision.models import MobileNetV2, Inception3

from torchvision.models import resnet18, resnet34


# 分割  不需要预训练模型


class MobileNetV2Define(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = MobileNetV2(num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):  # ji cheng
    def __init__(self):
        super(MLP, self).__init__()

        # zu zhuang

        # linear = nn.Linear(767*1022*3, 8)
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 256),  # 2 ^ n
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):  # image
        x = x.reshape(-1, 32 * 32 * 3)  # N V  5 32*32*3
        x = self.layers(x)  # out = linear(image)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 让继承起作用
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.ReLU(),  # 激活函数  负值变为0 正数保持不变
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, 1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.linear = nn.Linear(16 * 6 * 6, 8)  # N V

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(-1, 16 * 6 * 6)
        print(x.shape)
        x = self.linear(x)
        return x


# 残差网络
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # pretrained=True  预训练的模型
        # pretrained=Flase  不使用预训练模型
        self.resnet = resnet18(pretrained=False)
        self.linear = nn.Linear(1000, 8)

    def forward(self, x):
        x = self.resnet(x)
        x = x.reshape(-1, 1000)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    # Tensor  H W C  ->  C H W  -> N V
    inputs = torch.randn(5, 3, 32, 32)  # image N C H W
    net = ResNet18()
    # print(net)
    out = net(inputs)
    print(out)
    print(out.shape)

    # resnet = resnet18(pretrained=True)
    # print(resnet)
    # print(resnet.fc)
    #
    # resnet.fc = nn.Linear(512, 8)
    # print(resnet)





