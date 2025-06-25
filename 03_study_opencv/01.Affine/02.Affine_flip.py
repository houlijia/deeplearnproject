import cv2

image = cv2.imread('../00.image/lena512color.tiff')

image_01 = cv2.flip(image, 0)  # 垂直旋转
image_02 = cv2.flip(image, 1)  # 水平旋转
image_03 = cv2.flip(image, -1)  # 水平和垂直

cv2.imshow('垂直', image_01)
cv2.waitKey(0)
cv2.imshow('水平', image_02)
cv2.waitKey(0)
cv2.imshow('水平和垂直', image_03)
cv2.waitKey(0)
cv2.destroyAllWindows()
