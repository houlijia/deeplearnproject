import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = ["Heiti TC"]  # 设置中文


def plot_image_histogram(image_path, is_color):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.title("原始图像")
    origin_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(origin_gray)
    plt.axis('off')  # 去掉坐标

    plt.subplot(122)
    plt.title('hist')
    hist = cv2.calcHist([origin_gray], [0], None, [256], [0, 256])
    plt.plot(hist, color='black')  # hist的线为black
    plt.xlabel("像素值")
    plt.ylabel('像素数')
    plt.show()


if __name__ == '__main__':
    image_path = './image/hist_image.jpg'
    plot_image_histogram(image_path, True)
