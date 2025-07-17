import cv2
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../00.image/lena512color.tiff')
w, h, c = image.shape

pointSrc = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # 原始图像中 4点坐标
pointDst = np.float32([[int(w / 3), int(h / 3)], [int(w * 2 / 3), int(h / 3)], [0, h], [w, h]])  # 变换图像中 4点坐标

MP = cv2.getPerspectiveTransform(pointSrc, pointDst)

imgP =cv2.warpPerspective(image, MP, (h, w), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_WRAP)

plt.figure(figsize=(9, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original"), plt.axis('off')
plt.subplot(122), plt.imshow(cv2.cvtColor(imgP, cv2.COLOR_BGR2RGB)), plt.title("Projective"), plt.axis('off')
plt.show()
