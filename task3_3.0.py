import cv2 as cv
import numpy as np


# 旋转标靶

# 创建窗口
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.resizeWindow('img', 800, 600)
cv.namedWindow('res1', cv.WINDOW_NORMAL)
cv.resizeWindow('res1', 800, 600)
cv.namedWindow('res2', cv.WINDOW_NORMAL)
cv.resizeWindow('res2', 800, 600)
cv.namedWindow('res3', cv.WINDOW_NORMAL)
cv.resizeWindow('res3', 800, 600)

# 读图
img = cv.imread('2.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 颜色提取
low_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])
mask = cv.inRange(hsv, low_blue, upper_blue)

# 滤波
mask = cv.blur(mask, (5, 5))

# 轮廓识别
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(contours)

# 轮廓筛选
contour_max = max(contours, key=cv.contourArea)

# 取旋转矩形
cnt = contour_max
rect = cv.minAreaRect(cnt)
theta = rect[2]
# print(rect)
box = cv.boxPoints(rect)
box = np.int32(box)
cv.drawContours(img,[box],0,(0,0,255),2)

# 求旋转矩阵
# 获取图像的尺寸（行数和列数）
rows, cols = img.shape[:2]
place = [rows, cols]
# 旋转角度
angle = theta
# 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 0.6)
# 应用仿射变换（旋转）
res1 = cv.warpAffine(img, M, (cols, rows))
# 旋转每个轮廓的点
contour_max1 = cv.transform(contour_max, M)

# 判断是横着还是竖着,长>宽是竖着
(len, wid) = rect[1]
# print(len, wid)
if len < wid:
    # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    # 应用仿射变换（旋转）
    res2 = cv.warpAffine(res1, M, (cols, rows))
    # 旋转每个轮廓的点
    contour_max2 = cv.transform(contour_max1, M)
else:
    # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    # 应用仿射变换（旋转）
    res2 = cv.warpAffine(res1, M, (cols, rows))
    # 旋转每个轮廓的点
    contour_max2 = cv.transform(contour_max1, M)

# 判断正向还是逆向
top = contour_max2[contour_max2[:, :, 1].argmin()][0]  # 最上点
bottom = contour_max2[contour_max2[:, :, 1].argmax()][0]  # 最下点
# print(f'上{top}')
# print(f'下{bottom}')
# print(f'重心{place}')
top_y = int(top[1])
bottom_y = int(bottom[1])
# 求质心
M = cv.moments(contour_max2, True)
# print(M['m00'])
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
# 求距离
d_top = cy - top_y
d_down = bottom_y - cy
# 如果top>down则为正不用旋转
if d_top <= d_down:
    # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    # 应用仿射变换（旋转）
    res3 = cv.warpAffine(res2, M, (cols, rows))
else:
    # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    # 应用仿射变换（旋转）
    res3 = cv.warpAffine(res2, M, (cols, rows))


# 输出
cv.imshow('img', img)
cv.imshow('res1', res1)
cv.imshow('res2', res2)
cv.imshow('res3', res3)
cv.waitKey(0)
cv.destroyAllWindows()