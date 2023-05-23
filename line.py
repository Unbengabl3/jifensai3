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

def is_red_color(img, x1, y1, x2, y2):
    # 裁剪图像，提取给定区域
    region = img[y1:y2, x1:x2]
    if np.all(region == 255):
        # 处理区域全部为白色的情况
        return False
    else:
        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围
        lower_red1 = (0, 25, 25)
        upper_red1 = (10, 255, 255)
        lower_red2 = (160, 25, 25)
        upper_red2 = (180, 255, 255)

        # 利用inRange函数得到红色像素的掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 计算红色像素所占比例
        total_pixels = red_mask.shape[0] * red_mask.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        red_ratio = red_pixels / total_pixels

        # 判断红色像素所占比例是否大于0.3
        if red_ratio > 0.3:
            return True
        else:
            return False

def rho_theta_draw(src: np.ndarray, rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

    ptx1 = int((x1+x2) / 2) - 10
    ptx2 = int((x1 + x2) / 2) + 10
    pty = 300

    if (is_green_color(src, ptx1 - 10, pty - 10, ptx1 + 10, pty + 10) is True):
        cv2.putText(src, "Green", (ptx1, pty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        print(ptx1, pty)
        cv2.rectangle(src, (ptx1 - 10, pty - 10), (ptx1 + 10, pty + 10), (255, 0, 0), 2)

    if (is_green_color(src, ptx2 - 10, pty - 10, ptx2 + 10, pty + 10) is True):
        cv2.putText(src, "Green", (ptx2, pty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        print(ptx2, pty)
        cv2.rectangle(src, (ptx2 - 10, pty - 10), (ptx2 + 10, pty + 10), (255, 0, 0), 2)

    if (is_red_color(src, ptx1 - 10, pty - 10, ptx1 + 10, pty + 10) is True):
        cv2.putText(src, "Red", (ptx1, pty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        print(ptx1, pty)
        cv2.rectangle(src, (ptx1 - 10, pty - 10), (ptx1 + 10, pty + 10), (255, 0, 0), 2)

    if (is_red_color(src, ptx2 - 10, pty - 10, ptx2 + 10, pty + 10) is True):
        cv2.putText(src, "Red", (ptx2, pty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        print(ptx2, pty)
        cv2.rectangle(src, (ptx2 - 10, pty - 10), (ptx2 + 10, pty + 10), (255, 0, 0), 2)

def hough_draw(lines: np.ndarray, src: np.ndarray, min_rho, min_theta):
    drew_list = []  # 创建空列表用于存储已经作图的theta
    i = 0
    j = 0
    draw_flag = 1
    # 检查lines是否为空，否则作出第一条线
    if len(lines) != 0:
        rho, theta = lines[0, 0]  # 读取第一条直线数据

        rho_theta_draw(src, rho, theta)
        i += 1
        drew_list.append({'theta': theta, 'rho': rho})  # 在drew_list中存入第一个直线数据

    for rho, theta in lines[1:, 0]:  # lines是一个三维深度的数组，此处遍历每个[rho, theta]元素
        angle_threshold = np.pi / 18  # 10度作为阈值
        if (np.abs(theta - 0) > angle_threshold):
            continue
        for past_line in drew_list:
            theta_error = abs(past_line['theta'] - theta)
            rho_error = abs(past_line['rho'] - rho)
            if theta_error <= min_theta:  # 若两条直线的theta差值小于阈值
                if rho_error <= min_rho:  # 若两条直线距离rho小于阈值，则视作一条直线
                    j += 1
                    draw_flag = 0
                    break  # 跳出遍历
        else:  # 若函数正常完成遍历，说明是一条新直线
            drew_list.append({'theta': theta, 'rho': rho})  # 存入drew_list

        # 画图
        if draw_flag == 1:
            rho_theta_draw(src, rho, theta)
            i += 1
        else:
            draw_flag = 1  # 不作图，但恢复标志位

        pre_rho = rho  # 保存参数
        pre_theta = theta
    print(drew_list)

    print(f'输入{i+j}条直线，共作{i}条直线，{j}条被合并')
    return src



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(img_blur, 80, 150, apertureSize=3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)  # 直线检测
    if lines is not None:
        img = hough_draw(lines, frame, 80, np.pi / 5)
    else:
        img = frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()