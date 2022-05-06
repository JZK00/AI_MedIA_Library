#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from src.data_loader import MyDataset
from src.model import MobileNetV2Define
from src.utils import create_dir, dice_coef, get_date
import config as cfg
from torch.utils.tensorboard import SummaryWriter

from src.model import MLP, CNN, ResNet18

# 存放损失
log_dir = f"logs/logs-{get_date()}"
os.makedirs(log_dir, exist_ok=True)

summaryWriter = SummaryWriter(log_dir)


def train(epochs: int, batch_size: int, load_model_path: str = None, interval: int = 10):
    """
    :param epochs: 轮次
    :param batch_size: 批次
    :param load_model_path: 如果需要在原来的参数上继续训练, 指明保存的模型地址, 例如  load_model_path='net.pth'
    :param interval: 保存模型的间隔, 如果interval=10, 的每隔10轮保存一次模型
    """

    if len(cfg.CUDA_DEVICES) > 0:
        device = torch.device('cuda')  # GPU
    else:
        device = torch.device('cpu')

    model_dir = os.path.join(os.getcwd(), 'saved_models')
    create_dir(model_dir)

    # ######### dataset and dataloader
    train_dataset = MyDataset(cfg.TRAIN_TXT, cfg.IMAGE_DIR, cfg.SIZE, is_train=True)
    test_dataset = MyDataset(cfg.TEST_TXT, cfg.IMAGE_DIR, cfg.SIZE, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # net = MobileNetV2Define(len(cfg.LABEL_DICT)).to(device)

    # net = MLP().to(device)# 全连接
    # net = CNN().to(device)  # 卷积
    net = ResNet18().to(device)  # 卷积

    # ######### load model
    if load_model_path is not None:
        net.load_state_dict(torch.load(load_model_path))
        print(f'load model: {load_model_path}')
    else:
        print('no model')

    if len(cfg.CUDA_DEVICES) >= 2:
        net = nn.DataParallel(net).to(device)

    optimizer = optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        print('\nepoch:', epoch)

        train_loss = []
        test_loss = []

        # ######### train
        net.train()
        for i, (image, label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)

            # forward + backward + optimize
            out = net(image)

            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            print(f"epoch:{epoch} -> {i}/{len(train_dataloader)} -> train loss: {loss.item()}")

        if epoch % interval == 0:
            model_path = os.path.join(model_dir, f"net-{get_date()}_{epoch}.pth")

            if len(cfg.CUDA_DEVICES) >= 2:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)

            print(f'save model: {model_path}')


        # ######### test datasets
        net.eval()
        total = 0
        correct = 0
        for j, (image, label) in enumerate(test_dataloader):
            image = image.to(device)

            out = net(image).detach().cpu()
            loss = loss_fn(out, label)
            test_loss.append(loss.item())

            total += image.shape[0]

            out = torch.softmax(out, dim=1)
            pre = torch.argmax(out, dim=1)
            correct += torch.sum(torch.eq(pre, label)).item()

        # ######### calculate average loss in train datasets and test datasets
        # ######### calculate average dice coefficient in test datasets
        average_train_loss = np.array(train_loss).mean()
        average_test_loss = np.array(test_loss).mean()
        acc = correct / total

        summaryWriter.add_scalars('loss', {'train': average_train_loss, 'test': average_test_loss},
                                  global_step=epoch)
        summaryWriter.add_scalar('accuracy', acc, global_step=epoch)

        print(f"average train loss: {average_train_loss}, average test loss: {average_test_loss}, "
              f"accuracy: {acc}")


if __name__ == '__main__':
    train(epochs=10000, batch_size=4,
          load_model_path=None,
          interval=10)
