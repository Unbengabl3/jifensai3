import cv2
import numpy as np

def circle(img2):
    h, w = img2.shape[:2]
    # print(h, w)

    # 进行高斯模糊
    img_blur = cv2.GaussianBlur(img2, (5, 5), 0)
    cv2.imshow("blur", img_blur)

    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的幅值和方向
    mag, ang = cv2.cartToPolar(sobelx, sobely)

    # 设定阈值，提取出强梯度
    threshold = 50
    mag_thresh = np.zeros_like(mag)
    mag_thresh[mag > threshold] = mag[mag > threshold]

    # 对提取出的梯度进行二值化处理
    _, binary = cv2.threshold(mag_thresh, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', binary)
    binary = cv2.convertScaleAbs(binary)
    img_gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gyay', img_gray)

    # 设定圆形检测的参数
    min_radius = 40
    max_radius = 400
    dp = 1
    param1 = 100
    param2 = 70

    # 进行圆形检测
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp, minDist=20, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    # 绘制检测到的圆形
    if circles is not None:
        circles = sorted(np.round(circles[0, :]).astype("int"), key=lambda circle: circle[1])

        # 去除同心圆
        i = 0
        while i < len(circles) - 1:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[i + 1]
            if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) < 2 * r1:
                circles.pop(i + 1)
            else:
                i += 1

        # 绘制检测到的圆形
        for (x, y, r) in circles:
            cv2.circle(img2, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img2, (x, y), 2, (0, 0, 255), 3)

        for i, (x, y, r) in enumerate(circles):
            if (i == 0):
                souce = [0, 0]
                souce[0] = (x - w / 2) * 100 / w
                souce[1] = (y - h / 2) * 100 / h
                print(souce[0], souce[1])
            if ( i >= 1):
                continue


cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    circle(frame)
    cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()