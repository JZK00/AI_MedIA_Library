#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import numpy as np
import SimpleITK as sitk

"""
由于每个患者CT层数不一致，因此先把所有患者的CT合并之后，再进行训练
"""


def get_array(image_dir: str, label_dir: str, save_dir):
    """
    把同一个文件夹中所有患者的CT数据，合并到一起
    :param image_dir:
    :param label_dir:
    :param save_dir: 保存路径
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)

    image_all = []
    label_all = []

    for image_file in os.listdir(image_dir):
        # 读取图像
        image_path = os.path.join(image_dir, image_file)
        print(image_path)
        assert os.path.exists(image_path), f"{image_path}不存在"
        itk_img = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(itk_img)
        image_all.append(image)

        # 读取标签
        label_path = os.path.join(label_dir, f"segmentation-{image_file.split('-')[-1].split('.')[0]}.nii")
        print(label_path)
        assert os.path.exists(label_path), f"{label_path}不存在"
        itk_label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(itk_label)
        label[label > 0] = 1  # 本身有两个类别1和2，但只分割肝脏
        label_all.append(label)

    image_cat = np.concatenate(image_all, axis=0)  # 拼接特征
    label_cat = np.concatenate(label_all, axis=0)  # 拼接标签
    print("image_cat:", image_cat.shape)
    print("label_cat:", label_cat.shape)

    # 保存为numpy格式
    np.save(os.path.join(save_dir, 'image.npy'), image_cat)
    np.save(os.path.join(save_dir, 'label.npy'), label_cat)
    print(f'save:', save_dir)


if __name__ == '__main__':

    # 需要转换的目录：train validate
    dir_type = "validate"

    get_array(image_dir=f"../../datasets/LiTS/{dir_type}/image",
              label_dir=f"../../datasets/LiTS/{dir_type}/label",
              save_dir=f'./data/{dir_type}')
