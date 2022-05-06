import nrrd
import matplotlib.pyplot as plt

img_path = r"../../datasets/demo_NRRD/brain_image.nrrd"

data, header = nrrd.read(img_path)  # 数据  头文件
print(header)
print(data.shape)

for i in range(data.shape[2]):  # 循环每一层
    # 显示
    plt.imshow(data[..., i], cmap="gray")
    plt.show()
