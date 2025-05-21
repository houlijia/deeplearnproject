import cv2
import numpy as np

noise_img = np.random.normal(0, 200, (512, 512, 3)).astype(np.float32)
cv2.imshow('noise', noise_img)
cv2.waitKey(0)

lena = cv2.imread('./image/lena512color.tiff', 0)
# 均值滤波
blur_img = cv2.blur(lena, (3, 3))
cv2.imshow('blur img', blur_img)
cv2.waitKey(0)

# 方框滤波
# 当可选参数 normalize 为 True 的时候，方框滤波就是均值滤波，上式中的 a 就等于 1/9；normalize 为 False 的时候，a=1，相当于求区域内的像素和。
boxFilter_img = cv2.boxFilter(lena, -1, (3, 3), normalize=True)
cv2.imshow('boxFilter img', boxFilter_img, )
cv2.waitKey(0)

# 高斯滤波
gauss_img = cv2.GaussianBlur(lena, (3, 3), 1)  # 参数3: σx 值越大, 模糊效果越明显
cv2.imshow('gauss filter img', gauss_img)
cv2.waitKey(0)

# 中值滤波
median_img = cv2.medianBlur(lena, 3)
cv2.imshow('median Blur img', median_img)
cv2.waitKey(0)

# 双边滤波
bilateralFiler_img = cv2.bilateralFilter(lena, 9, 75, 75)
cv2.imshow('bilateralFiler img', bilateralFiler_img)
cv2.waitKey(0)
