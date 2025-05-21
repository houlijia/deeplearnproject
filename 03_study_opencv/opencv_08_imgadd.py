import cv2
import numpy as np

x = np.uint8([250])
print(f'x = {x}')
y = np.uint8([10])
print(f'y = {y}')
print(cv2.add(x, y))
print(x + y)  # 250 + 10 =260 % 256 = 4

img1 = cv2.imread('./image/lena512color.tiff', 0)
cv2.imshow('origin', img1)
cv2.waitKey(0)
noise = np.random.rand(512, 512)
cv2.imshow('noise', noise)
cv2.waitKey(0)
img2 = cv2.imread('./image/sibianxing.jpg', 0)
# cv2.imshow('other img', img2)
# cv2.waitKey(0)
new_img = cv2.addWeighted(img1, 0.5, noise, 0.5, 0, dtype=cv2.CV_8U)
cv2.imshow("new img", new_img)
cv2.waitKey(0)
