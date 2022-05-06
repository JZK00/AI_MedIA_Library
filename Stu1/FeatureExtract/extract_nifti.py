import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

data_dir = r"../../datasets/demo_NIFTI"  # 存放图像和标签的目录
img_path = os.path.join(data_dir, 'PANCREAS_0001.nii.gz')  # 图像
label_path = os.path.join(data_dir, 'label0001.nii.gz')  # 标签 ROI

# 特征保存位置 excel文件
save_path = os.path.join(data_dir, "feature.xlsx")

# 读取图像
itk_img = sitk.ReadImage(img_path)
img = sitk.GetArrayFromImage(itk_img)

# 读取标签
itk_label = sitk.ReadImage(label_path)
label = sitk.GetArrayFromImage(itk_label)

# 特征提取
extractor = featureextractor.RadiomicsFeatureExtractor()
feature = extractor.execute(sitk.GetImageFromArray(img), sitk.GetImageFromArray(label))

# 遍历每个特征
for name, value in feature.items():
    print(f"{name} : {value}")

# 保存特征
df = pd.DataFrame({k: str(v) for k, v in feature.items()}, index=[0])
df.to_excel(save_path, index=False)
print('save successfully.')
