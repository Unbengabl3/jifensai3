import cv2
import numpy as np
def is_green_color(img, x1, y1, x2, y2):
    # 裁剪图像，提取给定区域
    region = img[y1:y2, x1:x2]
    if np.all(region == 255):
        # Handle the case where region is all white
        return False
    else:
        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # 定义绿色的HSV范围
        lower_green = (36, 25, 25)
        upper_green = (70, 255, 255)

        # 利用inRange函数得到绿色像素的掩码
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 计算绿色像素所占比例
        total_pixels = mask.shape[0] * mask.shape[1]
        green_pixels = cv2.countNonZero(mask)
        green_ratio = green_pixels / total_pixels

        # 判断绿色像素所占比例是否大于0.3
        if green_ratio > 0.3:
            return True
        else:
            return False

def opencv_img(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)  # 直线检测
    if lines is None:
        return []

    # 将直线拟合成二维平面上的直线
    best_lines = []
    distance_threshold = 50

    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta

        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
            # 该直线与最后一行的焦点
            pt2 = (int((rho - img.shape[0] * np.sin(theta)) / np.cos(theta)), img.shape[0])
        else:  # 水平直线
            pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
            # 该直线与最后一列的交点
            pt2 = (img.shape[1], int((rho - img.shape[1] * np.cos(theta)) / np.sin(theta)))

        x1 = pt1[0]
        x2 = pt2[0]
        y1 = pt1[1]
        y2 = pt2[1]

        # 直线与垂直方向角度评定
        angle_threshold = np.pi / 18  # 10度作为阈值
        if (np.abs(theta - 0) > angle_threshold):
            continue



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lines = opencv_img(frame)

    if len(lines) >= 2:
        for i in range(len(lines)):
            line = lines[i]
            x1, y1, x2, y2 = line
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            print(pt1, pt2)

    # print((pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2)
    #
    # ptx = int((pt1[0]+pt2[0])/2)
    # pty = int((pt1[1]+pt2[1])/2)
    #
    # cv2.circle(frame, (ptx, pty), 10, (0, 255, 0), 2)
    # if(is_green_color(frame,ptx-5,pty-5,ptx+5,pty+5) is True):
    #     cv2.putText(frame, "Green", (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()