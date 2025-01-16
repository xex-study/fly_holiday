# 读取视频和摄像头画面并且使用imshow显示在屏幕上

import cv2 as cv
import numpy as np


cap = cv.VideoCapture('task_videos/task4_level1.mov') # 调用视频

while(1):
    ret, frame = cap.read() # 读取画面

    if ret == False:
        print('播放结束或未成功读取到视频')
        break
    else:
        cv.imshow('video', frame) # 显示画面
        if cv.waitKey(1) & 0xFF == ord('q'): # 按q结束
            break

cap.release()
cv.destroyAllWindows()

