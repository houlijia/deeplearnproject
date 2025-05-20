import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1、图像读取
img = cv2.imread('./image/lena512color.tiff')
px = img[100, 100]
print(f"100, 100 点的像素值为: {px}")

px = img[100, 100, 0]  # 只获取B通道值
print(f'100, 100 点的B通道值: {px}')
print(f'100, 100 点的G通道值: {img[100, 100, 1]}')
print(f'100, 100 点的R通道值: {img[100, 100, 2]}')
print(f"图片形状：{img.shape}")
print(f"图片的数据类型：{img.dtype}")
print(f"图片的像素数: {img.size}")  # 512*512*3

img_rio = img[50:450, 50:450, 2]  # 2表示R通道
cv2.imshow("RIO", img_rio)
cv2.waitKey(0)

# 2、image split
b, g, r = cv2.split(img)
cv2.imshow('B image', b)
cv2.waitKey(0)
cv2.imshow('G image', g)
cv2.waitKey(0)
cv2.imshow('R image', r)
cv2.waitKey(0)

# 3、image merge
img_merge = cv2.merge((b, g, r))
cv2.imshow('merge image', img_merge)
cv2.waitKey(0)

# 4、转换成灰度
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(f"所有的颜色转换模式：\n{flags}")
for (index, flag) in enumerate(flags):
    print(f"当前第{index+1}个转换模式为：{flag}")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray image", img_gray)
cv2.waitKey(0)


# 5、颜色空间转换
capture = cv2.VideoCapture('image/video_01.mp4')
# 蓝色的范围，不同光照条件下不一样，可灵活调整
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])

while True:
    # 1、捕获视频中一帧
    retBool, frame = capture.read()
    # 2、从BGR 转换到 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 3、inRange() , 介于 lower和 upper 之间为白色，其余为黑色
    masker = cv2.inRange(frame, lower_blue, upper_blue)
    # 4.只保留原图中的蓝色部分
    res = cv2.bitwise_and(frame, frame, mask=masker)
    cv2.imshow("frame", frame)
    cv2.imshow("hsv", hsv)
    cv2.imshow("range", res)
    if cv2.waitKey(1) == ord('q'):
        break


# 6、阈值分割
# ret, thresh = cv2.threshold(src, thresh, maxval, type)
# 参数：src：输入图像（单通道，如灰度图）。thresh：阈值（通常为 0-255 之间的数值）。maxval：当像素值超过阈值时的设定值（通常为 255）。 type：阈值类型（见下方表格）
# cv2.THRESH_BINARY	超过阈值的像素设为maxval，否则设为 0。
# cv2.THRESH_BINARY_INV	超过阈值的像素设为 0，否则设为maxval（与上一种相反）。
# cv2.THRESH_TRUNC	超过阈值的像素设为阈值，否则保持原值。
# cv2.THRESH_TOZERO	超过阈值的像素保持原值，否则设为 0。
# cv2.THRESH_TOZERO_INV	超过阈值的像素设为 0，否则保持原值。
# cv2.THRESH_OTSU	自动计算最优阈值（需与其他类型结合使用，如cv2.THRESH_BINARY + cv2.THRESH_OTSU）。
# 返回值：ret：实际使用的阈值（某些自适应方法会自动计算阈值）。thresh：处理后的二值图像。
img = cv2.imread('./image/lena512color.tiff', 0)  # 灰度图像
ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print(ret)
cv2.imshow('threshold', th)
cv2.waitKey(0)

# 7、几何变换
# opencv函数：cv2.resize()、 cv2.flip()、cv2.warpAffine()
img = cv2.imread('./image/lena512color.tiff', 0)
# 按照指定宽度和高度，缩放图片
res = cv2.resize(img, (256, 256))
# 按照比例缩放, 如 x, y 轴 均放大一倍
res2 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # interpolation插值方法: 默认是INTER_LINEAR
cv2.imshow("lena", img), cv2.imshow("res", res), cv2.imshow("res2", res2)
cv2.waitKey(0)

# 8、旋转图片
img = cv2.imread('./image/lena512color.tiff')
# 参数2 = 0：垂直翻转(沿 x 轴)，参数 2 > 0: 水平翻转(沿 y 轴)，参数 2 < 0: 水平垂直翻转
image_flip = cv2.flip(img, 0)
cv2.imshow("flip image", image_flip)
cv2.waitKey(0)

# 9、平移图片
img = cv2.imread('./image/lena512color.tiff', 1)
# 获取图像形状 shape = img.shape  # 返回 (高度, 宽度, 通道数) 或 (高度, 宽度)，是一个元组
row, cols = img.shape[:2]  # 切片操作，取元组的前两个元素（即高度和宽度）
# 定义平移矩阵，需要是 numpy 的 float32 类型
# x 轴平移 100，y 轴平移 50
M = np.float32([[1, 0, 100], [0, 1, 50]])
# 用仿射变换实现平移
# dst = cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
# src：输入图像（单通道或多通道）、M：2×3 的变换矩阵（数据类型为float32）、dsize：输出图像的大小 (宽度, 高度)。
# flags（可选）：插值方法（如cv2.INTER_LINEAR或cv2.INTER_NEAREST）、borderMode（可选）：边界填充方式（如cv2.BORDER_CONSTANT）、
# borderValue（可选）：边界填充值（默认 0，即黑色）
dst = cv2.warpAffine(img, M, (cols, row))
cv2.imshow("shift", dst)
cv2.waitKey(0)

# 10、旋转图片
img = cv2.imread('./image/lena512color.tiff', 1)
row, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, row / 2), 45, 0.5)  # cv2.getRotationMatrix2D() 旋转函数
dst = cv2.warpAffine(img, M, (cols, row))
cv2.imshow("rotation", dst)
cv2.waitKey(0)

# 11、绘图功能
# 创建一幅黑色图片
img = np.zeros((512, 512, 3), np.float32)
# 画一条线宽为5 的蓝色直线, 参数2为起点，参数3为终点
# cv2.line(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
# img：输入图像（会被直接修改）、pt1：起点坐标 (x1, y1)（元组或列表）、pt2：终点坐标 (x2, y2)、color：线条颜色（BGR 格式，如 (255, 0, 0) 为蓝色）。
# thickness（可选）：线条粗细（默认 1，负值表示填充，如 cv2.FILLED）、lineType（可选）：线条类型（如 cv2.LINE_AA 为抗锯齿线）。
# shift（可选）：坐标点的小数点位数（默认为 0）。
cv2.line(img, (0, 0), (512, 512), (255, 0, 0), 5)  # 所有绘图函数均会直接影响原图片，这点要注意，(255, 0, 0)表示线条为为蓝色
cv2.imshow("img", img)
cv2.waitKey(0)

# 画一个绿色边框的矩形，参数 2：左上角坐标，参数 3：右下角坐标
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
cv2.imshow("img", img)
cv2.waitKey(0)

# 画一个填充红色的圆，参数 2：圆心坐标，参数 3：半径
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
cv2.imshow("img", img)
cv2.waitKey(0)

# 在图中心画一个填充的半圆
# 第二个参数椭圆中心，第三个参数：x/y轴的长度，第四个参数：椭圆的旋转角度，第五个参数：椭圆的起始角度，第六个参数：椭圆的结束角度
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255, 0, 0), -1)
cv2.imshow("img", img)
cv2.waitKey(0)

# 定义四个顶点坐标
pts = np.array([[10, 5],  [50, 10], [70, 20], [20, 30]], np.int32)
# 顶点个数：4，矩阵变成 4*1*2 维
pts = pts.reshape((-1, 1, 2))
# 参数 3 如果是 False 的话，多边形就不闭合
cv2.polylines(img, [pts], True, (0, 255, 255))  # 如果需要绘制多条直线，使用 cv2.polylines() 要比 cv2.line() 高效很多
cv2.imshow("img", img)
cv2.waitKey(0)

# 添加文字
# 参数 2：要添加的文本
# 参数 3：文字的起始坐标（左下角为起点）
# 参数 4：字体
# 参数 5：文字大小（缩放比例）
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'ex2tron', (10, 500), font,
            4, (255, 255, 255), 2, lineType=cv2.LINE_AA)
cv2.imshow("img", img)
cv2.waitKey(0)

