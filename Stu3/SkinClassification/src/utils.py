#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import numpy as np
from datetime import datetime


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def dice_coef(output: np.ndarray, target: np.ndarray, axis=None):
    """output为预测结果 target为真实结果"""
    smooth = 1e-8  # 防止0除
    intersection = (output * target).sum(axis=axis)
    return (2. * intersection) / (output.sum(axis=axis) + target.sum(axis=axis) + smooth)


def get_date():
    """Get the current date"""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
