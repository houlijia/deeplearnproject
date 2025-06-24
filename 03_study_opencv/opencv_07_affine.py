# 仿射
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

img = cv2.imread('./00.image/sibianxing.jpg')
rows, cols = img.shape[:2]

# 变换前的三个点
pts1 = np.float32([[50, 65], [150, 65], [210, 210]])
pts2 = np.float32([[50, 100], [150, 65], [100, 250]])

# 生成变换矩阵
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
plt.figure(2)
plt.subplot(121), plt.imshow(img), plt.axis('off'), plt.title('origin')
plt.subplot(122), plt.imshow(dst), plt.axis('off'), plt.title('output')
plt.axis('off')
plt.show()
