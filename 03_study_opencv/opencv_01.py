import cv2
# https://codec.wang/docs/opencv/start/open-camera

# 1、读取图片
img = cv2.imread('./00.image/lena512color.tiff', 0)
# 参数 2：读入方式，省略即采用默认值
# cv2.IMREAD_COLOR：彩色图，默认值(1)
# cv2.IMREAD_GRAYSCALE：灰度图(0)
# cv2.IMREAD_UNCHANGED：包含透明通道的彩色图(-1)
# cv2.imshow("lena", img)
# cv2.waitKey(0)  # 0表示一直等待

# cv2.namedWindow("lena2", cv2.WINDOW_NORMAL)
# cv2.imshow("lena2", img)
# cv2.waitKey(0)

# 2、打开摄像头
# 参数 0 指的是摄像头的编号，如果你电脑上有两个摄像头的话，访问第 2 个摄像头就可以传入 1
# capture = cv2.VideoCapture(0)
#
# while True:
#     # 捕获一帧
#     # 函数返回的第1个参数,ret(return value缩写) 是一个布尔值，表示当前这一帧是否获取正确。cv2.cvtColor()
#     # 用来转换颜色，这里将彩色图转成灰度图
#     ret, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("capture image", gray)
#     if cv2.waitKey(1) == ord('q'):
#         break


# 3、播放本地视频
# capture = cv2.VideoCapture('./image/video_01.mp4')
# while capture.isOpened():
#     ret, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(5) == ord('q'):
#         break

# 4、录制视频
capture = cv2.VideoCapture(0)
furcc = cv2.VideoWriter_fourcc(*'mp4v')
outfile = cv2.VideoWriter('./image/output.avi', furcc, 255., (640, 480))

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        outfile.write(frame)
        cv2.waitKey(0)
    else:
        break