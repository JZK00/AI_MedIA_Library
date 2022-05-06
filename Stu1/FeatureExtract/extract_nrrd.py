import os
from radiomics import featureextractor
import pandas as pd

data_dir = r"../../datasets/demo_NRRD"  # 存放图像和标签的目录
img_path = os.path.join(data_dir, 'brain_image.nrrd')  # 图像
label_path = os.path.join(data_dir, 'brain_label.nrrd')  # 标签 ROI

# 特征保存位置 excel文件
save_path = os.path.join(data_dir, "feature.xlsx")

# 特征提取
extractor = featureextractor.RadiomicsFeatureExtractor()
feature = extractor.execute(img_path, label_path)

# 遍历每个特征
for name, value in feature.items():
    print(f"{name} : {value}")

# 保存特征
df = pd.DataFrame({k: str(v) for k, v in feature.items()}, index=[0])
df.to_excel(save_path, index=False)
print('save successfully.')
