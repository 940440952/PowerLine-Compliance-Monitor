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

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def se_resnet50(num_classes: int, pretrained_path=None):
    model = resnet50()
    for name, module in model.named_children():
        if name == 'layer1' or name == 'layer2' or name == 'layer3' or name == 'layer4':
            for basic_block in module:
                channels = basic_block.conv2.out_channels
                basic_block.add_module("se_block", SEBlock(channels))
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))

    model.eval()
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = se_resnet50(num_classes=6, pretrained_path='se_resnet50_model_85.pth')
model.to(device)

# 加载训练好的模型参数
ymodel = YOLO('yoloresult.pt')  # pretrained YOLOv8n model
model.eval()  # 将模型设置为评估模式


def predict(model, images_tensor):
    images_tensor = images_tensor.to(device)
    # 进行预测
    with torch.no_grad():
        outputs = model(images_tensor)
        predictions = torch.sigmoid(outputs)  # 应用sigmoid函数
    return predictions


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = {0: '高空作业人员', 1: '绝缘子', 2: '绝缘子', 3: '绝缘子'}


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

                # 根据 YOLO 识别的类别进行不同处理
                if object_class_id == 0:
                    # 正常流程，进行违规判别
                    # cropped_image = image.crop((x1, y1, x2, y2))
                    images_tensor = transform(image)
                    images_tensor = images_tensor.unsqueeze(0).to(device)
                    predictions = predict(model, images_tensor).squeeze(0)
                    value = torch.where(predictions > 0.4, torch.ones_like(predictions),
                                        torch.zeros_like(predictions))

                    violation_types = []
                    if value[0] == 1:
                        violation_types.append('安全带低挂高用')
                    if value[1] == 1:
                        violation_types.append('安全带和速差带未系')
                    if value[2] == 1:
                        violation_types.append('正常')
                    if value[3] == 1:
                        violation_types.append('速差未锁止')
                    if value[4] == 1:
                        violation_types.append('速差位置佩戴错误')
                    if value[5] == 1:
                        violation_types.append('速差低挂高用')
                    # violation_confidence = {
                    #     '速插式安全带位置错误': round(predictions[0].item(), 2),
                    #     '速插式安全带松弛': round(predictions[1].item(), 2),
                    #     '速插式安全带低挂高用': round(predictions[2].item(), 2),
                    #     '速插式安全带未系': round(predictions[3].item(), 2),
                    #     '腰部安全带未系': round(predictions[4].item(), 2),
                    #     '腰部安全带低挂高用': round(predictions[5].item(), 2),
                    # }
                else:
                    # 根据类别直接判定违规类型
                    violation_types = {
                        1: ["绝缘子未短接"],
                        2: ["正常"],
                        3: ["被遮挡或无法识别"]
                    }.get(object_class_id, ["未知"])

                violation_str = ", ".join(violation_types)
                # 根据违规类型选择边界框颜色
                red_violations = ['绝缘子未短接', '安全带低挂高用', '安全带和速差带未系', '速差未锁止',
                                  '速差位置佩戴错误', '速差低挂高用']
                orange_violations = ['被遮挡或无法识别']
                green_violations = ['正常']

                # 检查是否存在任何红色违规类型
                if any(violation in red_violations for violation in violation_types):
                    color = (0, 0, 255)  # 红色
                    violation_type = "疑似违章"
                # 检查是否存在橙色违规类型
                elif any(violation in orange_violations for violation in violation_types):
                    color = (0, 165, 255)  # 橙色
                    violation_type = "无法判断"
                else:
                    color = (0, 255, 0)  # 绿色
                    violation_type = "正常"

                if violation_type in ["疑似违章", "存疑"]:
                    assigned_violations = violation_types
                else:
                    assigned_violations = []

                # 添加检测信息
                detection_info = {
                    'id': bbox_id,
                    'class_name': object_class_name,
                    'bbox': [x1, y1, x2, y2],
                    'violation_type': violation_type,
                    'violations': assigned_violations,
                    # 'violation_confidence':violation_confidence
                }
                detections.append(detection_info)
                # 在图像上画出边界框
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 在图像上画出边界框
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)

                # 使用 PIL 在边界框旁边添加中文文本
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 40)
                text = f'ID: {bbox_id} 类别: {object_class_name}, 违规: {violation_str}'
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
