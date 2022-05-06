import SimpleITK as sitk
import matplotlib.pyplot as plt

img_path = r"../../datasets/demo_NIFTI/PANCREAS_0001.nii.gz"

itk_img = sitk.ReadImage(img_path)
img = sitk.GetArrayFromImage(itk_img)
print(itk_img)
print(img.shape)  # 表示各个维度的切片数量

# 逐层显示
for i in range(img.shape[0]):
    plt.imshow(img[i], cmap='gray')
    plt.show()
