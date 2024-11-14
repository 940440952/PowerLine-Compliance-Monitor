import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, request
import datetime
import numpy as np
import pytz
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
import os
import platform

app = Flask(__name__)

# 配置日志记录
file_handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)

# 设置日志记录等级
app.logger.setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型参数
try:
    ymodel = YOLO('./model/yolov8_seg.pt')
    app.logger.info('Model loaded successfully.')
except Exception as e:
    app.logger.error(f'Error loading model: {e}', exc_info=True)
    raise

class_names = {
    0: '速插位置系错', 1: '速插松弛', 2: '速插低挂高用', 3: '速插未系',
    4: '安全带未系', 5: '安全带低挂高用', 6: '无违规', 7: '人'
}

def get_font():
    system = platform.system()
    if system == "Windows":
        return "./Font/simhei.ttf"
    elif system == "Linux":
        return "./Font/NotoSansSC-Regular.ttf"
    else:
        return None

def calculate_font_size(bbox_width, bbox_height):
    min_font_size = 10
    max_font_size = 40
    font_size = int(min(bbox_width, bbox_height) / 10)
    font_size = max(min_font_size, min(max_font_size, font_size))
    return font_size

def save_output_image(cv_image):
    # 创建输出文件夹
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'cv_image_{timestamp}.jpg')

    # 保存图像
    cv2.imwrite(output_file, cv_image)
    return output_file

@app.route('/v1/service/imageTask', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            raise ValueError("No file part")

        file = request.files['file']
        if file.filename == '':
            raise ValueError("No selected file")

        image = Image.open(file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = ymodel.predict(image, save=False, show_conf=False)
        app.logger.info('Model predict successfully.')
        detections = []
        violations_list = []
        cv_image = image.copy()

        bbox_id = 1
        for r in results:
            bbox_list = r.boxes.xyxy.tolist()
            predict_res = r.boxes.cls.cpu().numpy()
            conf_res = r.boxes.conf.tolist()

            for i, bbox in enumerate(bbox_list):
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                object_class_id = int(predict_res[i])
                object_class_name = class_names.get(object_class_id, '未知')
                object_confidence = conf_res[i]
                violations_list.append(object_class_name)

                detection_info = {
                    "score": round(object_confidence*100, 1),
                    "leftTopX": x1,
                    "leftTopY": y1,
                    "rightBottomX": x2,
                    "rightBottomY": y2
                }
                detections.append(detection_info)

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                font_size = calculate_font_size(bbox_width, bbox_height)

                color = (0, 0, 255)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

                # Draw text with adaptive font size
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(get_font(), font_size)
                text = f'ID: {bbox_id} 类别: {object_class_name}'

                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = x1 + (bbox_width - text_width) / 2
                text_y = y1 - text_height - 5
                if text_y < 0:
                    text_y = y1 + 5
                draw.text((text_x, text_y), text, fill=color[::-1], font=font)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                bbox_id += 1

        # 保存输出图像
        saved_image_path = save_output_image(cv_image)

        tz = pytz.timezone('Asia/Shanghai')
        beijing_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            'resultCode': 200,
            'resultValue': {
                'analyseResults': "速插或安全带使用不规范" if any(item in violations_list for item in
                                                                   ['速插位置系错', '速插松弛', '速插低挂高用',
                                                                    '速插未系', '安全带未系', '安全带低挂高用']
                                                                   ) else "正常",
                'analyseTime': beijing_time,
                'outputImagePath': saved_image_path,  # 返回保存的图像路径
                'resultDetail': {
                    #'algCode': '81001001',
                    'resultDesc': violations_list,
                    'num': len(detections),
                    'resultItems': detections
                }
            },
            #"resultHint": None
        })

    except ValueError as e:
        app.logger.error(f'ValueError: {e}', exc_info=True)
        return jsonify({'resultCode': 400, 'message': str(e)}), 400

    except Exception as e:
        app.logger.error(f'Unexpected error: {e}', exc_info=True)
        return jsonify({'resultCode': 500, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)
