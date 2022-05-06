#!usr/bin/env python3
# -*- coding: UTF-8 -*-

# ######### 图片和标签两个文件夹中的文件名称, 须一一对应 #########
# 训练集目录
TRAIN_DIR = './data/train'

# 验证集目录
VALIDATE_DIR = './data/validate'

# 测试集图像目录
TEST_IMAGE_DIR = r"../../datasets/LiTS/test/image"
# 测试集标签目录
TEST_LABEL_DIR = r"../../datasets/LiTS/test/label"
# 分割结果保存目录  预测的结果
TEST_SEG_DIR = r"../../datasets/LiTS/test/seg"

# ######### 传入模型图片的宽和高, 通常宽高相等, 医学图像分割, 建议使用原图尺寸, 不进行缩放 #########
# SIZE = (512, 512)
SIZE = (16, 16)  # 教学用


# ######### 选择网络模型 #########
# 小模型 u2netp
# 大模型 u2net
MODEL_NAME = 'u2netp'


# ######### 选择训练的GPU型号 #########
# 如果没有GPU, 设置  CUDA_DEVICES = ""
# 假设电脑有3张显卡，且编号分别是 GPU0， GPU1， GPU2 。 如果使用三张卡训练，可设置 CUDA_DEVICES = "0, 1, 2"
CUDA_DEVICES = ""
