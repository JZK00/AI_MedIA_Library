#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import random
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config as cfg



def tran(image):
    return transforms.Compose([
        transforms.ToTensor(),  # 0-1
        transforms.Normalize((0.5,), (0.5,)),  # -1  -   1
    ])(image)


def rotate_img(image):
    random_value = random.randint(0, 4)

    if random_value == 0:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif random_value == 1:
        rotate_code = cv2.ROTATE_180
    elif random_value == 2:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return image

    image = cv2.rotate(image, rotate_code)

    return image


def online_enhance(image):
    image = rotate_img(image)
    return image


class MyDataset(Dataset):
    def __init__(self, file_path, file_dir, size, is_train=True):
        self.file_path = file_path
        self.file_dir = file_dir
        self.size = size
        self.is_train = is_train

        with open(self.file_path, 'r') as fr:
            self.all_files = fr.readlines()

    def __len__(self):
        return len(self.all_files)

    #
    def __getitem__(self, index):
        # ######### read image
        strs = self.all_files[index]
        # print(strs)
        strs = str(strs).strip()

        image_name, label_txt = strs.split('\t')
        image_path = os.path.join(self.file_dir, image_name)
        image = cv2.imread(image_path)

        if len(image.shape) == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) > 3:
            image = image[..., 0:3]

        if self.is_train:
            image = online_enhance(image)  # 样本增强

        image = cv2.resize(image, self.size)  # 改变尺寸
        img = tran(image)  # 归一化、标准化

        label = cfg.LABEL_DICT[label_txt]
        label = torch.tensor(label, dtype=torch.int64)

        return img, label

