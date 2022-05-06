#!usr/bin/env python3
# -*- coding: UTF-8 -*-

# 图片存放目录  不要有中文
IMAGE_DIR = r'../../datasets/skin_classification/ori_img'

# 训练集的文本路径
TRAIN_TXT = 'data/train.txt'

# 验证集的文本路径
TEST_TXT = 'data/val.txt'

# 分类
LABEL_DICT = {
    'MEL': 0,
    'NV': 1,
    'BCC': 2,
    'AK': 3,
    'BKL': 4,
    'DF': 5,
    'VASC': 6,
    'SCC': 7
}

# ######### 传入模型图片的宽和高, 通常宽高相等#########
SIZE = (32, 32)

# ######### 选择训练的GPU型号 #########
# 如果没有GPU, 设置  CUDA_DEVICES = ""

# 假设电脑有3张显卡，且编号分别是 GPU0， GPU1， GPU2
# 如果使用三张卡训练，可设置 CUDA_DEVICES = "0, 1, 2"

CUDA_DEVICES = ""
