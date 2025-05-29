import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

lena_img = cv2.imread('image/lena512color.tiff', 0)
lena_img = cv2.cvtColor(lena_img, cv2.COLOR_BGR2RGB)
lena_face = cv2.imread('./image/lena_face.jpg', 0)
lena_face = cv2.cvtColor(lena_face, cv2.COLOR_BGR2RGB)
lena_face_h, lena_face_c = lena_face.shape[:2]

res = cv2.matchTemplate(lena_img, lena_face, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
left_top = max_loc
right_botton = (left_top[0] + lena_face_c, left_top[1] + lena_face_h)
processed_image = cv2.rectangle(lena_img, left_top, right_botton, 255, 2)
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('template image')
plt.imshow(lena_face)
plt.axis('off')
plt.subplot(122)
plt.title('target image')
plt.axis('off')
plt.imshow(processed_image)
plt.show()
