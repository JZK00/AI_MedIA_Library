import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pydicom

img_dir = r"../../datasets/demo_DICOM/emphysema"

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)

    info = pydicom.dcmread(img_path)  # 读取Dicom文件信息说明
    print('info:\n', info)

    # 获取Dicom数据
    ds = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(ds)
    print('img_array:\n', img_array)
    print("shape:", img_array.shape)

    # 显示
    plt.imshow(img_array[0], cmap="gray")
    plt.show()
