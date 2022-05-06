#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def tran(image):
    """
    归一化 标准化
    :param image:
    :return:
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])(image)


def rotate_img(image, label):
    """
    旋转图像
    :param image:
    :param label:
    :return:
    """
    random_value = random.randint(0, 4)

    if random_value == 0:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif random_value == 1:
        rotate_code = cv2.ROTATE_180
    elif random_value == 2:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return image, label

    image = cv2.rotate(image, rotate_code)
    label = cv2.rotate(label, rotate_code)

    return image, label


def crop_img(image, label):
    """
    裁剪图像
    :param image:
    :param label:
    :return:
    """

    random_value = random.randint(0, 2)
    if random_value > 0:
        return image, label

    h, w = image.shape[:2]

    left_x_range = random.randint(0, int(w / 5))
    left_y_range = random.randint(0, int(h / 5))
    right_x_range = random.randint(int(w / 5 * 4), w)
    right_y_range = random.randint(int(h / 5) * 4, h)

    # crop image and label
    crop_image = image[left_y_range:right_y_range, left_x_range:right_x_range]
    crop_label = label[left_y_range:right_y_range, left_x_range:right_x_range]

    # restore size by filling zero
    temp_image = np.zeros(image.shape, dtype=np.uint8)
    temp_label = np.zeros(label.shape, dtype=np.uint8)

    new_h, new_w = crop_image.shape[:2]
    difference_h = h - new_h
    difference_w = w - new_w

    temp_image[difference_h // 2:difference_h // 2 + new_h, difference_w // 2:difference_w // 2 + new_w] = crop_image
    temp_label[difference_h // 2:difference_h // 2 + new_h, difference_w // 2:difference_w // 2 + new_w] = crop_label

    return temp_image, temp_label


def online_enhance(image, label):
    """
    在线增强，只针对训练集而言
    :param image:
    :param label:
    :return:
    """
    image, label = rotate_img(image, label)
    image, label = crop_img(image, label)
    return image, label


class SalObjDataset(Dataset):
    def __init__(self, dir_path, size: tuple, is_train: bool = True):
        """
        :param dir_path: 目录
        :param size: 传入模型的图片尺寸
        :param is_train: 如果为True，进行数据增强，用于训练；为False, 不进行数据增强，用于测试
        """
        self._size = size
        self._is_train = is_train

        self._image = np.load(os.path.join(dir_path, "image.npy"))
        self._label = np.load(os.path.join(dir_path, "label.npy"))

    def __len__(self):
        return self._image.shape[0]

    def __getitem__(self, index):
        image = self._image[index]
        label = self._label[index]

        # cv2.imshow('img', image)
        # cv2.imshow('label', (label * 255).astype(np.uint8))

        # data online enhance
        if self._is_train:
            image, label = online_enhance(image, label)

        # cv2.imshow('img enhance', image)
        # cv2.imshow('label enhance', (label * 255).astype(np.uint8))

        # ######### resize the image and label
        image = cv2.resize(image, self._size, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, self._size, interpolation=cv2.INTER_NEAREST)

        # cv2.imshow('img resize', image)
        # cv2.imshow('label resize', (label * 255).astype(np.uint8))
        # cv2.waitKey()

        image = tran(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label


if __name__ == '__main__':
    s = SalObjDataset(r"data/train", size=(64, 64))

    while True:
        i, j = s[300]
        print(i.shape, j.shape)
