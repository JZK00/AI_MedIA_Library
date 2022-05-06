import cv2

img_path = r"../../datasets/demo_other/Image_01L.jpg"

img = cv2.imread(img_path)  # 读取图片
print(img.shape)

# 显示
cv2.imshow('img', img)
cv2.waitKey()
