import os
import pydicom

img_dir = r"../../datasets/demo_DICOM/emphysema"  # 原数据目录

save_dir = r"../../datasets/demo_DICOM/emphysema_mask"  # 存储脱敏的dicom文件目录
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建

for img_name in os.listdir(img_dir):
    print("img name:", img_name)
    img_path = os.path.join(img_dir, img_name)

    ds = pydicom.dcmread(img_path)  # 读取Dicom文件信息说明
    print('ds:\n', ds)

    # 数据脱敏
    ds.PatientName = "xxx"  # 姓名
    ds.PatientID = "0000"  # id

    print("ds mask:\n", ds)

    # 保存脱敏后的数据
    ds.save_as(os.path.join(save_dir, img_name))
