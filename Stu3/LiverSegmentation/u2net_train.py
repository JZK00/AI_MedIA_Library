#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from data_loader import SalObjDataset
from u2net import U2NET, U2NETP
from utils import model_evaluate, get_date
import config as cfg
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_DEVICES

# 存放损失
log_dir = f"logs/logs-{get_date()}"
os.makedirs(log_dir, exist_ok=True)

summaryWriter = SummaryWriter(log_dir)

bce_loss = nn.BCELoss()


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


def train(epochs: int, batch_size: int, load_model_path: str = None, interval: int = 1):
    """
    :param epochs: 轮次
    :param batch_size: 批次
    :param load_model_path: 如果需要在原来的参数上继续训练, 指明保存的模型地址, 例如  load_model_path='net.pth'
    :param interval: 保存模型的间隔, 如果interval=1, 即每1轮保存一次模型。如果interval=10,即每10轮保存一次模型
    """

    if len(cfg.CUDA_DEVICES) > 0:
        device = torch.device('cuda')  # GPU
    else:
        device = torch.device('cpu')

    model_name = cfg.MODEL_NAME
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    os.makedirs(model_dir, exist_ok=True)

    # ######### dataset and dataloader
    train_dataset = SalObjDataset(cfg.TRAIN_DIR, cfg.SIZE, is_train=True)
    val_dataset = SalObjDataset(cfg.VALIDATE_DIR, cfg.SIZE, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ######### define the net
    if model_name == 'u2net':
        net = U2NET(1, 1).to(device)
    elif model_name == 'u2netp':
        net = U2NETP(1, 1).to(device)
    else:
        raise ValueError("MODEL_NAME setup error")

    print(f"The chosen network model is '{model_name}'")

    # ######### load model
    if load_model_path is not None:
        net.load_state_dict(torch.load(load_model_path))
        print(f'load model: {load_model_path}')
    else:
        print('no model')

    if len(cfg.CUDA_DEVICES) >= 2:
        net = nn.DataParallel(net).to(device)

    optimizer = optim.Adam(net.parameters())

    for epoch in range(1, epochs + 1):
        print('\nepoch:', epoch)

        train_loss = []
        val_loss = []
        DSC_val = []
        SEN_val = []
        IOU_val = []

        # ######### train
        net.train()
        for i, (image, label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(image)

            loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label.unsqueeze(1))

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向求导
            optimizer.step()  # 参数更新

            train_loss.append(loss.item())

            if i % 10 == 0:
                print(f"epoch:{epoch} -> {i}/{len(train_dataloader)} -> train loss: {loss.item()}")

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss0, loss, image, label

        if epoch % interval == 0:
            model_path = os.path.join(model_dir, model_name + f"-{epoch}.pth")

            if len(cfg.CUDA_DEVICES) >= 2:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)

            print(f'save model: {model_path}')

        # ######### validate datasets
        net.eval()
        for image, label in val_dataloader:
            image = image.to(device)
            label = label.to(device)

            d0, d1, d2, d3, d4, d5, d6 = net(image)
            loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label.unsqueeze(1))

            val_loss.append(loss.item())

            # 在验证集上进行模型评估
            output = d0.cpu().detach().squeeze(1).numpy()
            label = label.cpu().detach().numpy()
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            for idx in range(label.shape[0]):
                temp_output = output[idx]
                temp_label = label[idx]
                if temp_label.sum() > 0:  # 标签存在ROI区域才进行评估
                    DSC, SEN, IOU = model_evaluate(temp_output, temp_label)
                    DSC_val.append(DSC)
                    SEN_val.append(SEN)
                    IOU_val.append(IOU)

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, image, label

        # ######### calculate average loss in train datasets and val datasets
        # ######### model evaluate in val datasets
        average_train_loss = np.array(train_loss).mean()
        average_val_loss = np.array(val_loss).mean()
        average_DSC_val = np.array(DSC_val).mean()
        average_SEN_val = np.array(SEN_val).mean()
        average_IOU_val = np.array(IOU_val).mean()

        summaryWriter.add_scalars('loss', {'train': average_train_loss, 'validate': average_val_loss},
                                  global_step=epoch)
        summaryWriter.add_scalars('evaluate', {"DSC": average_DSC_val, "SEN": average_SEN_val, "IOU": average_IOU_val},
                                  global_step=epoch)

        print(f"average train loss: {average_train_loss}, average val loss: {average_val_loss}, "
              "validate evaluate:", {"DSC": average_DSC_val, "SEN": average_SEN_val, "IOU": average_IOU_val})


if __name__ == '__main__':
    train(epochs=100, batch_size=6,
          load_model_path=None,
          interval=1)
