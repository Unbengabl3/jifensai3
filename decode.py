import cv2
import numpy as np
from pyzbar import pyzbar

def decode_qr_code(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 执行二维码解码
    decoded_objects = pyzbar.decode(gray)

    # 打印解码结果
    for obj in decoded_objects:
        data = obj.data.decode("utf-8")
        print(obj.type, data)

    # 在图像上绘制解码结果
    for obj in decoded_objects:
        # 提取二维码的边界框坐标
        x, y, w, h = obj.rect

        # 在图像上绘制边界框和标签
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    decode_qr_code(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()