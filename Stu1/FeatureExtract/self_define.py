import os
from radiomics import featureextractor
import pandas as pd

"""
自定义提取特征
"""

data_dir = r"../../datasets/demo_NRRD"  # 存放图像和标签的目录
img_path = os.path.join(data_dir, 'brain_image.nrrd')  # 图像
label_path = os.path.join(data_dir, 'brain_label.nrrd')  # 标签 ROI

# 特征保存位置 excel文件
save_path = os.path.join(data_dir, "feature_define.xlsx")

# 初始化特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor()

# 图像类型
extractor.disableAllImageTypes()  # 禁用所有图像类型

extractor.enableImageTypeByName('Wavelet')  # 小波滤波

# sigma：浮点数或整数列表，必须大于 0。用于高斯核的滤波器宽度 (mm)（确定粗糙度）
# extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1, 2]})  # 拉普拉斯-高斯滤波

# 个别特征提取
extractor.disableAllFeatures()  # 禁用所有特征
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glcm')

# 特征提取
feature = extractor.execute(img_path, label_path)

# 遍历每个特征
for name, value in feature.items():
    print(f"{name} : {value}")

# 保存特征
df = pd.DataFrame({k: str(v) for k, v in feature.items()}, index=[0])
df.to_excel(save_path, index=False)
print('save successfully.')
