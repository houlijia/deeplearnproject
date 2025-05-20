import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as pli

# origin img
new_img = cv2.imread('./image/lena512color.tiff')
# bmp
cv2.imwrite('img_bmp.bmp', new_img)  # 文件大小：359KB

# jpg 默认 95% 质量
cv2.imwrite('img_jpg95.jpg', new_img)  # 文件大小：52.3KB
# jpg 20% 质量
cv2.imwrite('img_jpg20.jpg', new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 20])  # 文件大小：8.01KB
# jpg 100% 质量
cv2.imwrite('img_jpg100.jpg', new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 文件大小：82.5KB

# png 默认 1 压缩比
cv2.imwrite('img_png1.png', new_img)  # 文件大小：240KB
# png 9 压缩比
cv2.imwrite('img_png9.png', new_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])  # 文件大小：207KB

# Matplotlib
# 显示灰度图
img = cv2.imread('./image/lena512color.tiff', 0)
# 灰度图显示，cmap(color map) 设置为 gray
plt.figure(1)
plt.imshow(img, cmap='gray')
plt.title('cv2 img gray')
plt.pause(2)

# 显示彩色图
img = cv2.imread('./image/lena512color.tiff')
# 或使用
# img2 = img[:, :, ::-1]
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示不正确的图
plt.figure(2)
plt.subplot(121), plt.imshow(img)

# 显示正确的图
plt.subplot(122)
plt.xticks([]), plt.yticks([])  # 隐藏 x 和 y 轴
plt.imshow(img2)
plt.pause(2)

# Matplotlib 读取图片
img = pli.imread('./image/lena512color.tiff')
plt.figure(3)
plt.imshow(img)
plt.savefig('./image/lena_matplotlib.jpg')
plt.show()
