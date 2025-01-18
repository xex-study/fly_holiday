import cv2 as cv
import numpy as np


cap = cv.VideoCapture('task_videos/task4_level3.mp4')
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('video_34.avi', fourcc, 20.0, (640, 480))

# 创建窗口
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 600, 480)
cv.namedWindow('rotate_res', cv.WINDOW_NORMAL)
cv.resizeWindow('rotate_res', 600, 480)
cv.namedWindow('count_res', cv.WINDOW_NORMAL)
cv.resizeWindow('count_res', 600, 480)



while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        print('读图结束或读图失败！')
        break

    # 颜色提取
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    low_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    mask = cv.inRange(hsv, low_blue, upper_blue)

    # 轮廓识别
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 判断contours是否为空
    if len(contours) == 0:
        continue
    else:
        # 轮廓筛选
        contour_max = max(contours, key=cv.contourArea)

        # 取旋转矩形
        cnt = contour_max
        rect = cv.minAreaRect(cnt)
        theta = rect[2]
        box = cv.boxPoints(rect)
        box = np.int32(box)

        # 求旋转矩阵（第一步旋转：把图像转为横向或纵向）
        # 获取图像的尺寸（行数和列数）
        rows, cols = frame.shape[:2]
        place = [rows, cols]
        # 旋转角度
        angle = theta
        # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 0.6)
        # 应用仿射变换（旋转）
        res1 = cv.warpAffine(frame, M, (cols, rows))
        # 旋转每个轮廓的点
        contour_max1 = cv.transform(contour_max, M)

        # 第二步旋转：将图像旋转致纵向
        # 判断是横着还是竖着,长>宽是竖着
        (leng, wid) = rect[1]
        # print(len, wid)
        if leng < wid:
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

        # 第三步旋转：将倒立的图像转为正向
        # 判断正向还是逆向
        top = contour_max2[contour_max2[:, :, 1].argmin()][0]  # 最上点
        bottom = contour_max2[contour_max2[:, :, 1].argmax()][0]  # 最下点
        top_y = int(top[1])
        bottom_y = int(bottom[1])
        # 求质心
        M = cv.moments(contour_max2, True)
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
            rotate_res = cv.warpAffine(res2, M, (cols, rows))
        else:
            # 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
            # 应用仿射变换（旋转）
            rotate_res = cv.warpAffine(res2, M, (cols, rows))
        # x, y, w, h = cv.boundingRect(contour_max1)
        # cv.rectangle(rotate_res, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # 已转正，提取数字部分ROI

        hsv = cv.cvtColor(rotate_res, cv.COLOR_BGR2HSV)

        # 提取蓝色部分，得到掩膜
        low_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        mask = cv.inRange(hsv, low_blue, upper_blue)

        # 滤波
        cv.GaussianBlur(mask, (5, 5), 0)

        # 轮廓识别
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # 判断
        if len(contours) == 0:
            continue
        else:
            # 轮廓筛选
            contour_max = max(contours, key=cv.contourArea)

            # 创建纯黑图版
            output = np.zeros_like(rotate_res)

            # 遍历轮廓并在黑色图像上绘制
            for contour in contours:
                cv.drawContours(output, [contour], -1, (255, 255, 255), thickness=cv.FILLED)
            img = cv.bitwise_and(rotate_res, output)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # 图像处理
            ing = cv.bitwise_not(mask)
            ing = cv.resize(ing, (img.shape[1], img.shape[0]))
            res = cv.bitwise_and(ing, gray)

            # 提取数字部分

            # 提取轮廓
            contours_count, _ = cv.findContours(res, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # 判断轮廓是否为空
            if len(contours_count) == 0:
                continue
            else:
                # 筛选轮廓
                ContourCount_max = max(contours_count, key=cv.contourArea)

                # 取边界矩形
                x, y, w, h = cv.boundingRect(ContourCount_max)
                count_res = res[y:y + h, x:x + w]


        # 输出结果
        cv.imshow('frame', frame)
        cv.imshow('rotate_res', rotate_res)
        cv.imshow('count_res', count_res)
        if cv.waitKey(15) & 0xFF == ord('q'):
            print('手动结束！')
            break

cap.release()
out.release()
cv.destroyAllWindows()