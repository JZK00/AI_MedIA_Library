#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from data_loader import tran
from u2net import U2NET, U2NETP
from utils import model_evaluate
import config as cfg
import SimpleITK as sitk
import cv2


def test(load_model_path: str, threshold: float = 0.5):
    """
    模型测试
    :param load_model_path:
    :param threshold:
    :return:
    """
    if len(cfg.CUDA_DEVICES) > 0:
        device = torch.device('cuda')  # GPU
    else:
        device = torch.device('cpu')

    model_name = cfg.MODEL_NAME
    # ######### define the net
    if model_name == 'u2net':
        net = U2NET(1, 1)
    elif model_name == 'u2netp':
        net = U2NETP(1, 1)
    else:
        raise ValueError("MODEL_NAME setup error")

    print(f"The chosen network model is '{model_name}'")

    # 加载模型
    net.load_state_dict(torch.load(load_model_path, map_location='cpu'))
    net = net.to(device)
    net.eval()

    df_evl = []  # 保存所有病例在测试集上的评估结果
    for image_file in os.listdir(cfg.TEST_IMAGE_DIR):
        # 读取图像
        image_path = os.path.join(cfg.TEST_IMAGE_DIR, image_file)
        print("image path:", image_path)
        assert os.path.exists(image_path), f"{image_path}不存在"
        itk_img = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(itk_img)
        print("image shape:", image.shape)

        # 读取标签
        label_name = f"segmentation-{image_file.split('-')[-1].split('.')[0]}.nii"
        label_path = os.path.join(cfg.TEST_LABEL_DIR, label_name)
        print("label path:", label_path)
        assert os.path.exists(label_path), f"{label_path}不存在"
        itk_label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(itk_label)
        label[label > 0] = 1  # 本身有两个类别1和2，但只分割肝脏
        print("label shape:", label.shape)

        # 每一层进行预测
        predict = []
        DSC_test = []
        SEN_test = []
        IOU_test = []
        for idx in range(image.shape[0]):
            img_slice = image[idx]
            label_slice = label[idx]

            img_slice = cv2.resize(img_slice, cfg.SIZE, interpolation=cv2.INTER_NEAREST)
            img_slice = tran(img_slice)
            img_slice = img_slice.unsqueeze(0)

            # 分割预测
            output = net(img_slice.to(device))[0].cpu().detach().squeeze().numpy()
            output[output > threshold] = 1
            output[output <= threshold] = 0
            output = cv2.resize(output, label_slice.shape, interpolation=cv2.INTER_NEAREST)  # 预测结果还原
            output = output.astype(np.uint8)
            predict.append(output[None, ...])

            # 模型评估
            if label_slice.sum() > 0:  # 标签存在ROI区域才进行评估
                DSC, SEN, IOU = model_evaluate(output, label_slice)
                DSC_test.append(DSC)
                SEN_test.append(SEN)
                IOU_test.append(IOU)

        # 保存评估结果
        DSC_test = np.array(DSC_test).mean()
        SEN_test = np.array(SEN_test).mean()
        IOU_test = np.array(IOU_test).mean()
        print("DSC:", DSC_test)
        print("SEN:", SEN_test)
        print("IOU:", IOU_test)

        df_evl.append([image_file, DSC_test, SEN_test, IOU_test])

        # 保存分割结果
        seg = np.concatenate(predict, axis=0)
        sitk_img = sitk.GetImageFromArray(seg)
        os.makedirs(cfg.TEST_SEG_DIR, exist_ok=True)
        sitk.WriteImage(sitk_img, os.path.join(cfg.TEST_SEG_DIR, label_name))

    df_evl = pd.DataFrame(df_evl, columns=["image", "DSC", "SEN", "IOU"])
    df_evl.to_excel(os.path.join(cfg.TEST_SEG_DIR, "evaluate.xlsx"), index=False)
    print(df_evl)


if __name__ == '__main__':
    test(load_model_path='saved_models/u2netp/u2netp-1.pth')
