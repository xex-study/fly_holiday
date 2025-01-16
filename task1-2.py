# 读取摄像头的视频并显示在屏幕上，将录制到的视频保存

import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0) # 读取摄像头
fourcc = cv.VideoWriter_fourcc(*'XVID') # 设定fourcc代码
out = cv.VideoWriter('outvideo.avi', fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == False:
        print('读取失败！')
        break
    else:
        out.write(frame)
        cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()