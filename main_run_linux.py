import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, request
import base64
import datetime
import numpy as np
import pytz
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
from io import BytesIO
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

def is_valid_base64_image(base64_str):
    try:
        # 解码 Base64 字符串
        image_data = base64.b64decode(base64_str)
        # 尝试从字节数据中创建图像
        image = Image.open(BytesIO(image_data))
        # 验证图像是否有效
        image.verify()  # 验证图像头部是否有效
        # 如果图像能够正常打开并且验证通过，则返回 True
        return True
    except (base64.binascii.Error, IOError, ValueError) as e:
        # 捕获解码错误、IO 错误或值错误
        return False
def calculate_font_size(bbox_width, bbox_height):
    # Define the minimum and maximum font size
    min_font_size = 10
    max_font_size = 40

    # Calculate the font size based on the width and height of the bounding box
    font_size = int(min(bbox_width, bbox_height) / 10)
    font_size = max(min_font_size, min(max_font_size, font_size))

    return font_size


@app.route('/v1/service/imageTask', methods=['POST'])
def upload():
    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data:
                raise ValueError("No data received")     //  请求为空

            if 'image' not in data:
                raise ValueError("No image part")        //  未传图片

            if 'algCode' not in data:
                raise ValueError("No algCode")          //  缺少 algCode

            alg_code = data['algCode']
            if alg_code != "81001001":
                raise ValueError("algCode value error")     //  algCode错误

            if 'analyseId' not in data:
                raise ValueError("No analyseId")      //   无分析id

            image_data = data['image']
            if is_valid_base64_image(image_data) is False:
                raise ValueError("Base64 image padding is invalid.")   //  图片解码错误

            image_bytes = base64.b64decode(image_data)
            image_np_arr = np.frombuffer(image_bytes, np.uint8)

            image = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR)

            results = ymodel.predict(image, save=False, show_conf=False)
            app.logger.info('Model predict successfully.')
            detections = []
            violations_list = []
            cv_image = np.array(image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

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
                        "score": round(object_confidence*100,1),
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
                    if text_y < 0:  # Ensure text does not go off the image
                        text_y = y1 + 5
                    draw.text((text_x, text_y), text, fill=color[::-1], font=font)
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    bbox_id += 1

            cv2.imwrite('cv_image.jpg', cv_image)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()


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
                    #'rawImageData': image_data,
                    #'osdImageData': img_str,
                    'resultDetail': {
                        'algCode': '81001001',
                        'resultDesc': violations_list,
                        'num': len(detections),
                        'resultItems': detections
                    }
                },
                "resultHint": None
            })

    except ValueError as e:
        app.logger.error(f'ValueError: {e}', exc_info=True)
        return jsonify({'resultCode': 400, 'message': str(e)}), 400

    except Exception as e:
        app.logger.error(f'Unexpected error: {e}', exc_info=True)
        return jsonify({'resultCode': 500, 'message': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)

# gunicorn -w 4 -b 0.0.0.0:8887 wsgi:app
