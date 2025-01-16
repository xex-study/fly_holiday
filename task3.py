import cv2 as cv
import numpy as np

# 旋转标靶

# 创建窗口
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.resizeWindow('res', 800, 600)
cv.namedWindow('dst', cv.WINDOW_NORMAL)
cv.resizeWindow('dst', 800, 600)
cv.namedWindow('dst1', cv.WINDOW_NORMAL)
cv.resizeWindow('dst1', 800, 600)

img = cv.imread('photo2.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 颜色提取
low_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])
mask = cv.inRange(hsv, low_blue, upper_blue)
# 特征识别
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(contours)
for contour in contours:
    if cv.contourArea(contour) > 200:
        M = cv.moments(contour)
        if M['m00'] != 0:
            #画出轮廓
            cv.drawContours(img, contours, 0, (0, 0, 255), 3)
        else:
            print('error')
    else:
        print('error')
# 取旋转矩形
cnt = contours[0]
rect = cv.minAreaRect(cnt)
theta = rect[2]
# print(theta)
box = cv.boxPoints(rect)
box = np.int32(box)
cv.drawContours(img,[box],0,(0,0,255),2)
# 求旋转矩阵
# 获取图像的尺寸（行数和列数）
rows, cols = img.shape[:2]
# 旋转角度
angle = (theta - 90)
# 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
# 应用仿射变换（旋转）
dst = cv.warpAffine(img, M, (cols, rows))
# 判断正向还是逆向
top = contours[0][contours[0][:, :, 1].argmin()][0]  # 最上点
bottom = contours[0][contours[0][:, :, 1].argmax()][0]  # 最下点
# print(top)
# print(f'下{bottom}')
top_y = int(top[1])
bottom_y = int(bottom[1])
# 求质心
M = cv.moments(contours[0], True)
print(M['m00'])
cy = int(M['m01'] / M['m00'])
# 求距离
d_top = cy - top_y
d_down = bottom_y - cy
# 如果top>down则为正不用旋转
if d_top >= d_down:
    # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    # 应用仿射变换（旋转）
    dst1 = cv.warpAffine(dst, M, (cols, rows))
else:
    # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    # 应用仿射变换（旋转）
    dst1 = cv.warpAffine(dst, M, (cols, rows))



cv.imshow('res', img)
cv.imshow('dst', mask)
cv.imshow('dst1', dst1)
cv.waitKey(0)
cv.destroyAllWindows()