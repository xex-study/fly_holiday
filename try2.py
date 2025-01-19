import cv2 as cv
import numpy as np
import os

# 创建一个保存图片的文件夹  
output_folder1 = 'res1'    
os.makedirs(output_folder1, exist_ok=True)
output_folder2 = 'res2'    
os.makedirs(output_folder2, exist_ok=True)
output_folder3 = 'numbers'    
os.makedirs(output_folder3, exist_ok=True)
res1_number = 0
res2_number = 0
frame_number = 0


# 蓝色范围 
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])


cap = cv.VideoCapture('photo/task_videos/task4_level1.mov')  


frame_count = 0  # 用来为保存的图像命名
save_frame_interval = 3  # 每5帧保存一次
save_count = 0  # 用于文件命名
save_count1 = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('读图结束!')
        break
    img = frame.copy()


    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # 滤波
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # 颜色提取
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 轮廓为空跳过当前帧
    if len(contours) == 0:
        continue

    for contour in contours:
        if cv.contourArea(contour) > 400: # 利用面积筛选轮廓 
            
            # 逼近轮廓为多边形
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            # 过滤非五边形轮廓
            if len(approx) != 5:
                continue

            # 计算轮廓的矩形框
            x, y, w, h = cv.boundingRect(contour)

            # 在原图像上绘制矩形框
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # 裁剪出蓝色区域
            res1_img = img[y:y + h, x:x + w]

            if res1_img.size < 400:  # 如果裁剪的图像太小，则跳过
                continue
            res1_imgs = []
            res1_imgs.append(res1_img)
            res1_number += 1
            cv.imwrite(os.path.join(output_folder1, f"res1_{res1_number}.png"), res1_img)
            for res1_img in res1_imgs:
                dst0 = res1_img.copy()
                gray0 = cv.cvtColor(dst0, cv.COLOR_BGR2GRAY)
                hsv0 = cv.cvtColor(dst0, cv.COLOR_BGR2HSV)
                mask_blue0 = cv.inRange(hsv0, lower_blue, upper_blue)
                mask_blue0 = cv.morphologyEx(mask_blue0, cv.MORPH_CLOSE, kernel)
                contours_blue, _ = cv.findContours(mask_blue0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                contour_max0 = max(contours_blue, key=cv.contourArea)
                M = cv.moments(contour_max0)
                  # 获取旋转矩形
                rect = cv.minAreaRect(contour_max0)
                theta = rect[2]
                box = cv.boxPoints(rect)
                box = np.int32(box)
                cv.drawContours(dst0, [box], 0, (0, 0, 255), 2)
                # 获取图像尺寸
                rows, cols = dst0.shape[:2]
                angle = theta  # 旋转角度.
                # 计算旋转矩阵并旋转图像
                M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                dst1 = cv.warpAffine(dst0, M, (cols, rows))
                contour_max1 = cv.transform(contour_max0, M)

                # 根据长宽比判断是否需要旋转
                (w, h) = rect[1]
                aspect_ratio = float(w) / h
                if aspect_ratio < 1:
                    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
                    dst2 = cv.warpAffine(dst1, M, (cols, rows))
                    contour_max2 = cv.transform(contour_max1, M)
                else:
                    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
                    dst2 = cv.warpAffine(dst1, M, (cols, rows))
                    contour_max2 = cv.transform(contour_max1, M)

                # 利用极点判断正逆向
                top = contour_max2[contour_max2[:, :, 1].argmin()][0]
                bottom = contour_max2[contour_max2[:, :, 1].argmax()][0]
                top_y = int(top[1])
                bottom_y = int(bottom[1])
                M = cv.moments(contour_max2, True)
                cy = int(M['m01'] / M['m00'])
                d_top = cy - top_y
                d_down = bottom_y - cy
                if d_top <= d_down:
                    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
                    dst3 = cv.warpAffine(dst2, M, (cols, rows))
                else:
                    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
                    dst3 = cv.warpAffine(dst2, M, (cols, rows))
                res2_number += 1
                cv.imwrite(os.path.join(output_folder2, f"res2_{res2_number}.png"), dst3)
            # 显示处理后的帧
                img1 = dst3.copy()
                gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
                hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
                mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
                blue_region = cv.bitwise_and(img1, img1, mask=mask_blue)
                gray = cv.cvtColor(blue_region, cv.COLOR_BGR2GRAY)

                # 二值化图像
                ret, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
                contours, _ = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                # 填充轮廓
                dst = cv.drawContours(dst, contours, -1, 255, cv.FILLED)

                # 将填充后的轮廓与掩模相减
                res = cv.absdiff(dst, mask_blue)
                res = cv.GaussianBlur(res, (5, 5), 0)
                kernel = np.ones((3, 3), np.uint8)
                res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel)
                num = cv.bitwise_and(gray1, gray1, mask=res)

                # 获取 num 图像中的轮廓
                contours_num, _ = cv.findContours(num, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                if len(contours_num) == 0:
                    continue  # 如果没有找到轮廓，则跳过

                # 找到最大的轮廓
                largest_contour = max(contours_num, key=cv.contourArea)

                # 逼近轮廓为多边形
                epsilon = 0.02 * cv.arcLength(largest_contour, True)
                approx = cv.approxPolyDP(largest_contour, epsilon, True)

                # 如果拟合的多边形顶点数小于4或者大于6，则跳过当前帧
                if len(approx) != 4:
                    continue

                # 获取轮廓的边界框
                x, y, w, h = cv.boundingRect(largest_contour)

                # 在原图像中裁剪出轮廓所在区域
                cropped_image_num = num[y:y+h, x:x+w]
                cv.imshow('img', cropped_image_num)

                if cropped_image_num.size < 400:  # 如果像素数小于400，则跳过
                    continue
                
                cv.imshow('res', cropped_image_num)
                cv.resizeWindow('res', 400,300)
                # 保存裁剪图像
                frame_number += 1
                cv.imwrite(os.path.join(output_folder3, f"number_{frame_number}.png"), cropped_image_num)

    
    # 按1毫秒延时显示图像，并按'q'退出
    cv.imshow('frame', frame)
    cv.resizeWindow('frame', 640,480)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv.destroyAllWindows()
