import cv2

image = cv2.imread('../00.image/lena512color.tiff')
w, h, c = image.shape
image_01 = cv2.resize(image, (int(0.75*w), int(h)))
cv2.imshow('resize', image_01)
cv2.waitKey(0)
cv2.destroyAllWindows()

