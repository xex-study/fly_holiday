# 检测出视频中的蓝色块，实时绘制出他们的轮廓和重心

import cv2 as cv
import numpy as np


t1 = cv.getTickCount()
cap = cv.VideoCapture('task_videos/task4_level1.mov')
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('result.avi', fourcc, 20.0, (640,480))

# 得到目标掩膜
target = cv.imread('target.png')
target1 = cv.imread('target1.png')
tar1_hsv = cv.cvtColor(target1, cv.COLOR_BGR2HSV)
tar_hsv = cv.cvtColor(target,cv.COLOR_BGR2HSV)
low_blue = np.array([110, 50, 50])
up_blue = np.array([130, 255, 255])
tar_mask = cv.inRange(tar_hsv, low_blue,up_blue)
tar_mask1 = cv.inRange(tar1_hsv, low_blue,up_blue)
tar_contours, tar_hierarchy = cv.findContours(tar_mask, cv.CHAIN_APPROX_SIMPLE, cv.RETR_TREE)
tar1_contours, tar_hierarchy1 = cv.findContours(tar_mask1, cv.CHAIN_APPROX_SIMPLE, cv.RETR_TREE)
area = cv.contourArea(tar_contours[0])
area1 = cv.contourArea(tar1_contours[0])
print(f'最大面积为{area}')
print(f'最小面积为{area1}')
res = cv.drawContours(target, tar_contours,0, (0, 255, 0), 3)
res1 = cv.drawContours(target1, tar1_contours,0, (0, 255, 0), 3)
# cv.imshow('res', res)
# cv.imshow('res1', res1)
# cv.waitKey(0)
# cv.destroyAllWindows()

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == False:
        print('播放结束！')
        break
    else:
        #转化为hsv提取颜色对象
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        low_blue = np.array([110, 50, 50])
        up_blue = np.array([130, 255, 255])
        mask = cv.inRange(hsv, low_blue,up_blue)

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # 转化为灰度图
        # 标准阈值化
        # ret, threshould = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        # #自适应二值化处理
        # ret, thre = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,10, 2)
        #轮廓检测
        contours, hierarchy = cv.findContours(mask, cv.CHAIN_APPROX_SIMPLE, cv.RETR_TREE)
        # 绘制出轮廓
        img = cv.drawContours(frame, contours, -1, (0, 255, 0),3)
        # 绘制重心
        for contour in contours:
            if cv.contourArea(contour) > 200:
                M = cv.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy =int(M['m01'] / M['m00'])
                    # 画出重心
                    cv.circle(img, (cx, cy), 3 , (0, 0, 255), 2)
        # 播放
        cv.imshow('frame', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
# out.release()
cv.destroyAllWindows()

t2 = cv.getTickCount()
time = int((t2 - t1)) / cv.getTickFrequency()
time = int(time)
print(f'花费的时间为{time}秒')

