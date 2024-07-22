import base64

from flask import Flask, request
import numpy as np
from torch import nn
from torchvision.models import resnet50
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
from flask import Flask, jsonify  # flask库
import time
from io import BytesIO
from torchvision.models import resnet50, ResNet50_Weights
app = Flask(__name__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型参数
ymodel = YOLO('yolov8_seg.pt')

# 0: sucha_position
# 1: sucha_slack
# 2: sucha_diguagaoyong
# 3: sucha_disconnect
# 4: safebelt_disconnect
# 5: safebelt_diguagaoyong
# 6: no_violation
# 7: person

class_names = {0: '速插位置系错', 1: '速插松弛', 2: '速插低挂高用', 3: '速插未系', 4: '安全带未系', 5: '安全带低挂高用', 6: '无违规', 7: '人'}


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(file).convert("RGB")
        start_time = time.time()

        results = ymodel.predict(image, save=False, show_conf=False)
        detections = []

        # OpenCV 处理图像
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        bbox_id = 1
        for r in results:
            bbox_list = r.boxes.xyxy.tolist()  # 获取边界框列表
            predict_res = r.boxes.cls.cpu().numpy()  # 获取类别预测结果

            for i, bbox in enumerate(bbox_list):
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                object_class_id = int(predict_res[i])  # 获取对象的类别ID
                object_class_name = class_names.get(object_class_id, '未知')

                # 添加检测信息
                detection_info = {
                    'id': bbox_id,
                    'class_name': object_class_name,
                    'bbox': [x1, y1, x2, y2],
                }
                detections.append(detection_info)
                # 在图像上画出边界框
                color = (0, 0, 255)  # 红色
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

                # 使用 PIL 在边界框旁边添加中文文本
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 40)
                text = f'ID: {bbox_id} 类别: {object_class_name}'
                draw.text((x1, y1 - 25), text, color[::-1], font=font)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                bbox_id += 1  # 递增 bbox ID



        # 将 OpenCV 图像转换回 PIL 图像
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # 将 PIL 图像转换为 base64 编码
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        end_time = time.time()
        inference_time = end_time - start_time

        return jsonify({
            'inference_time': inference_time,
            'marked_image_base64': img_str,
            'violations': detections
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
