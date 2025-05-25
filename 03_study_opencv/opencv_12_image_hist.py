import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('./image/hist_image.jpg', 0).astype(np.uint8)
cv2.imshow('hist origin image', img)
cv2.waitKey(0)
# 第一个参数是图片,用[], 第二个是通道, 彩色图 B/G/R 分别传入[0]/[1]/[2], 第三个是计算区域，计算整幅图的话，输入None
# 第四个参数是子区段数目，如果我们统计 0~255 每个像素值，bins=256；如果划分区间，比如 0~15, 16~31…240~255 这样 16 个区间，bins=16
# range:要计算的像素值范围，一般为[0,256)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure(1)
plt.plot(hist)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()


