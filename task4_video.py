import cv2 as cv
import numpy as np


cap = cv.VideoCapture('task_videos/task4_level3.mp4')
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('res4.avi', fourcc, 20.0, (640,480))


cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', (800, 600))
cv.namedWindow('mask', cv.WINDOW_NORMAL)
cv.resizeWindow('mask', (800, 600))
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.resizeWindow('img', (800, 600))
cv.namedWindow('ing', cv.WINDOW_NORMAL)
cv.resizeWindow('ing', (800, 600))
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.resizeWindow('res', (800, 600))

while(cap.isOpened()):
    # 读图，并转化到hsv色彩空间
    ret, frame = cap.read()
    if ret == False:
        print('读图结束！')
        break
    else:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # 提取蓝色部分，得到掩膜
        low_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        mask = cv.inRange(hsv, low_blue, upper_blue)

         # 滤波
        cv.GaussianBlur(mask,(5,5),0)

        # 轮廓识别
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        # 轮廓筛选
        if len(contours) == 0:
            continue
        else:
            contour_max = max(contours,key=cv.contourArea)

            # 创建纯黑图版
            output = np.zeros_like(frame)

            # 遍历轮廓并在黑色图像上绘制
            for contour in contours:
                cv.drawContours(output, [contour], -1, (255, 255, 255), thickness=cv.FILLED)
            img = cv.bitwise_and(frame, output)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # 图像处理
            ing = cv.bitwise_not(mask)
            ing = cv.resize(ing, (img.shape[1], img.shape[0]))
            res = cv.bitwise_and(ing, gray)

            # 输出
            cv.imshow('frame', frame)
            cv.imshow('mask', mask)
            cv.imshow('img', img)
            cv.imshow('ing', ing)
            cv.imshow('res', res)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
out.release()
cv.destroyAllWindows()
