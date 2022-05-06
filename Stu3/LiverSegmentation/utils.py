#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
from datetime import datetime


def get_date():
    """Get the current date"""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def model_evaluate(output: np.ndarray, target: np.ndarray):
    """
    如果标签中的值全是0，即没有ROU区域，则 TP = 0
    如果评估指标求平均，会导致结果变小

    Args:
        output: 预测结果，值只有0和1两类, shape: [H, W]
        target: 真实结果，值只有0和1两类, shaep: [H, W]

    Returns:
        DSC: Dice系数
        SEN: 敏感度
        IOU: 交并比
    """
    assert set(target.reshape(-1))

    TP = (output * target).sum()
    TN = ((1 - target) * (1 - output)).sum()
    FP = ((1 - target) * output).sum()
    FN = (target * (1 - output)).sum()

    DSC = 2 * TP / (FP + 2 * TP + FN + 1e-8)  # Dice相似系数(Dice Similariy Coefficient,DSC)
    SEN = TP / (TP + FN + 1e-8)  # Sensitivity 敏感度
    IOU = TP / (TP + FP + FN + 1e-8)  # 交并比

    return DSC, SEN, IOU


if __name__ == '__main__':
    output = np.array([[0, 0, 1, 0],
                       [0, 1, 1, 0],
                       [0, 1, 0, 1]])
    target = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0]])
    print(output)
    print(target)

    print(model_evaluate(output, target))
