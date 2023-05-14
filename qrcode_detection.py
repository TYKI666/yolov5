import torch
import cv2
import numpy as np
import serial
from models.experimental import attempt_load
from utils.datasets import letterbox

# 初始化串口通信
ser = serial.Serial("COM10", 9600)
# 加载二维码识别模型
net = cv2.dnn.readNetFromTensorflow('qrcode.pb')
# 加载yolov5模型
weights = 'D:/桌面/bysj/KINHA/4/yolov5-5.0/weights/yolov5s.pt'
model = attempt_load(weights, map_location=torch.device('cpu'))  # 指定cpu设备
# 指定输入分辨率
img_size = 640
# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧图像
    ret, frame = cap.read()
    # 缩放图像
    height, width, channels = frame.shape
    scale = img_size / max(height, width)
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    # 将图像转换为RGB格式
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 对图像进行letterbox预处理
    img = letterbox(frame, new_shape=img_size)[0]
    # 将图像转换为PyTorch Tensor格式
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    # 模型预测
    outputs = model(img)[0]
    # 对预测结果进行非极大值抑制（NMS）
    outputs = outputs.cpu().detach().numpy()
    outputs[:, :4] = outputs[:, :4] / scale
    bboxes = outputs[:, :4][outputs[:, 4] >= 0.5]
    scores = outputs[:, 4][outputs[:, 4] >= 0.5]
    # 遍历检测结果
    for bbox, score in zip(bboxes, scores):
        # 绘制边框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 提取二维码区域
        qrcode = frame[y1:y2, x1:x2]
        # 缩放二维码图像
        qrcode = cv2.resize(qrcode, (224, 224))
        # 预处理二维码图像
        qrcode = cv2.cvtColor(qrcode, cv2.COLOR_BGR2RGB)
        qrcode = qrcode / 255.0
        qrcode = (qrcode - 0.5) / 0.5
        # 将二维码图像传输给飞行控制程序
        qrcode_str = np.array2string(qrcode, separator=',', max_line_width=1000000)
        ser.write(qrcode_str.encode('utf-8'))
    # 显示图像
    cv2.imshow('frame', frame)
    # 检测键盘按键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头
cap.release()
# 关闭串口
ser.close()
# 销毁窗口
cv2.destroyAllWindows()