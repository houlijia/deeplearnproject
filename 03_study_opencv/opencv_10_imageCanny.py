import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


origin_img = cv2.imread('./image/canyoriginImage.jpg', 0)
cv2.imshow('origin', origin_img)
edges = cv2.Canny(origin_img, 30, 30)  # 参数1和2为最低和最高阈值
cv2.imshow('cany image', edges)
cv2.waitKey(0)


# 先阈值分割，再边缘检测
_, threshold_img = cv2.threshold(origin_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
cv2.imshow('边缘检测图片', threshold_img)
cv2.waitKey(0)

edges = cv2.Canny(origin_img, 30, 30)  # 参数1和2为最低和最高阈值
cv2.imshow('cany image', edges)
cv2.waitKey(0)

cv2.imshow('Cany new Img', np.hstack((origin_img, threshold_img, edges)))  # np.hstack水平堆叠
cv2.waitKey(0)
