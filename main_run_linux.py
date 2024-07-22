import base64
import datetime

import numpy as np
import pytz
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
from flask import Flask, jsonify, request  # flask库
import time
from io import BytesIO
from torchvision.models import resnet50, ResNet50_Weights
import platform

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型参数
ymodel = YOLO('./model/yolov8_seg.pt')

# 0: sucha_position
# 1: sucha_slack
# 2: sucha_diguagaoyong
# 3: sucha_disconnect
# 4: safebelt_disconnect
# 5: safebelt_diguagaoyong
# 6: no_violation
# 7: person

class_names = {0: '速插位置系错', 1: '速插松弛', 2: '速插低挂高用', 3: '速插未系', 4: '安全带未系', 5: '安全带低挂高用',
               6: '无违规', 7: '人'}


def get_font():
    system = platform.system()
    if system == "Windows":
        return "./Font/simhei.ttf"
    elif system == "Linux":
        return "./Font/NotoSansSC-Regular.ttf"
    else:
        return None


@app.route('/v1/service/imageTask', methods=['POST'])
def upload():
    if request.method == 'POST':
        data = request.get_json()
        if 'image' not in data:
            raise ValueError("No image part")  # 未上传图片或格式错误

        if 'algCode' not in data:
            raise ValueError("No algCode")  # 无算法编码

        alg_code = data['algCode']
        if alg_code != "81001001":
            raise ValueError("algCode value error")  # 算法编码错误

        if 'analyseId' not in data:
            raise ValueError("No analyseId")  # 无分析id

        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        image_np_arr = np.frombuffer(image_bytes, np.uint8)

        try:
            image = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Error decoding image")  # 无法读取该格式
        except Exception as e:
            raise ValueError(f"Error opening image: {str(e)}")

        start_time = time.time()

        results = ymodel.predict(image, save=False, show_conf=False)
        detections = []
        violations_list = []
        # OpenCV 处理图像
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        bbox_id = 1
        for r in results:
            bbox_list = r.boxes.xyxy.tolist()  # 获取边界框列表
            predict_res = r.boxes.cls.cpu().numpy()  # 获取类别预测结果
            conf_res = r.boxes.conf.tolist()

            for i, bbox in enumerate(bbox_list):
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                object_class_id = int(predict_res[i])  # 获取对象的类别ID
                object_class_name = class_names.get(object_class_id, '未知')
                object_confidence = conf_res[i]
                violations_list.append(object_class_name)

                # 添加检测信息
                detection_info = {
                    # 'id': bbox_id,
                    # 'class_name': object_class_name,
                    # 'bbox': [x1, y1, x2, y2],
                    "score": object_confidence,
                    "leftTopX": x1,
                    "leftTopY": y1,
                    "rightBottomX": x2,
                    "rightBottomY": y2
                }
                detections.append(detection_info)
                # 在图像上画出边界框
                color = (0, 0, 255)  # 红色
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

                # 使用 PIL 在边界框旁边添加中文文本
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(get_font(), 40)
                text = f'ID: {bbox_id} 类别: {object_class_name}'
                draw.text((x1, y1 - 25), text, color[::-1], font=font)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                bbox_id += 1  # 递增 bbox ID

        cv2.imwrite('cv_image.jpg', cv_image)
        # 将 OpenCV 图像转换回 PIL 图像
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # 将 PIL 图像转换为 base64 编码
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        end_time = time.time()
        inference_time = end_time - start_time

        tz = pytz.timezone('Asia/Shanghai')
        beijing_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        # return jsonify({
        #     'inference_time': inference_time,
        #     'marked_image_base64': img_str,
        #     'violations': detections
        # })
        return jsonify({
            'resultCode': 200,
            'resultValue': {
                'analyseResults': "速插或安全带使用不规范" if any(item in violations_list for item in
                                                                  ['速插位置系错', '速插松弛', '速插低挂高用',
                                                                   '速插未系', '安全带未系', '安全带低挂高用']
                                                                  ) else "正常",
                'analyseTime': beijing_time,
                'rawImageData': image_data,
                'osdImageData': img_str,
                'resultDetail':
                    {
                        'algCode': '81001001',
                        'resultDesc': violations_list,
                        'num': len(detections),
                        'resultItems': detections
                    }
            },
            "resultHint": None
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
