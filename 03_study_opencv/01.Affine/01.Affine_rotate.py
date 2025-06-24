import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../00.image/lena512color.tiff', flags=1)
image_R90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
image_R180 = cv2.rotate(image, cv2.ROTATE_180)
image_R270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

plt.figure(1)
plt.subplot(221), plt.axis('off'), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('origin')
plt.subplot(222), plt.axis('off'), plt.imshow(cv2.cvtColor(image_R90, cv2.COLOR_BGR2RGB)), plt.title('90')
plt.subplot(223), plt.axis('off'), plt.imshow(cv2.cvtColor(image_R180, cv2.COLOR_BGR2RGB)), plt.title('180')
plt.subplot(224), plt.axis('off'), plt.imshow(cv2.cvtColor(image_R270, cv2.COLOR_BGR2RGB)), plt.title('270')

plt.show()
