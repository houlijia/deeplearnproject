import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./image/noise.jpg', 0)
# 用法
# retval, dst = cv2.threshold(src, thresh, maxval, type)
# src:输入图像, thresh:设定的阈值, maxval:当前像素超过阈值时赋予的新值(通常为255)
# type: 阈值处理类型
# 固定阈值法
ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# Otsu 阈值法
ret2, th2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 先进行高斯滤波，再使用Otsu 阈值法
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2.imshow('ret1', th1)
cv2.imshow('ret2', th2)
cv2.imshow('ret3', th3)
cv2.waitKey(0)
