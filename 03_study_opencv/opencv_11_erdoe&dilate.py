import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./image/j_image.jpg', 0).astype(np.uint8)
cv2.imshow('j image', img)
cv2.waitKey(0)

# 腐蚀
# kernel = np.ones((5, 5), np.uint8)
# 可以用 cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)) 来生成不同形状的结构元素
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
erode_img = cv2.erode(img, kernel)
cv2.imshow('erode', erode_img)
cv2.waitKey(0)

# 膨胀、与腐蚀相反，取的是局部最大值, 效果是变胖
dilation_img = cv2.dilate(img, kernel)
cv2.imshow('dilation', dilation_img)
cv2.waitKey(0)


# 开运算：先腐蚀后膨胀叫开运算
opening_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening img', opening_img)
cv2.waitKey(0)

# 闭运算, 先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除/"闭合"物体里面的小黑洞，所以叫闭运算）
close_img = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
cv2.imshow('close img', close_img)
cv2.waitKey(0)
