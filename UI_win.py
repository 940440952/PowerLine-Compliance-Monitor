import tkinter as tk
from tkinter import filedialog, Label, PanedWindow, StringVar
from PIL import Image, ImageTk, ImageDraw, ImageFont
import requests


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        api_url = url_entry.get()
        response = send_image_to_api(file_path, api_url)
        if response:
            show_image_with_violations(file_path, response["violations"])
            show_violations_in_label(response["violations"])


def send_image_to_api(file_path, url):
    files = {'image': open(file_path, 'rb')}
    try:
        response = requests.post(url, files=files)
        return response.json()
    except requests.RequestException as e:
        print(f"请求错误: {e}")
        return None


def show_image_with_violations(file_path, violations):
    img = Image.open(file_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simsun.ttc", 20)
    except IOError:
        font = ImageFont.load_default()

    for v in violations:
        bbox = v['bbox']

        # 检查是否存在任何红色违规类型
        if v['violation_type'] == "疑似违章":
            color = (255, 0, 0)  # 红色
        # 检查是否存在橙色违规类型
        elif v['violation_type'] == "存疑":
            color = (255, 165, 0)  # 橙色
        # 检查是否'腰部安全带未系'和'速插式安全带未系'都出现
        elif v['violation_type'] == "正常":
            color = (0, 255, 0)  # 绿色
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=color, width=2)
        text = f"{v['id']} {v['class_name']}"
        text_position = (bbox[0], bbox[1] - 25 if bbox[1] - 25 > 0 else bbox[3])
        draw.text(text_position, text, fill=color, font=font)

    img.thumbnail((1280, 720))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk


def show_violations_in_label(violations):
    violations_text = "违章信息:\n"
    for v in violations:
        violations_text += f"ID: {v['id']}, 类型: {v['class_name']},是否违章:{v['violation_type']}, 违章: {', '.join(v['violations'])}\n"
    text_label.config(text=violations_text)


# 初始化窗口
root = tk.Tk()
root.title("违章内容检测(高空作业、绝缘子)")

# 设置固定窗口大小
root.geometry("1920x1080")

# 创建分割窗口
paned_window = PanedWindow(root, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=1)

# 创建左侧的Frame用于显示图片
left_frame = tk.Frame(paned_window, width=1280, height=720)
left_frame.pack_propagate(False)  # 防止内部组件影响Frame的大小
paned_window.add(left_frame)

# 用于显示图像的标签
img_label = Label(left_frame)
img_label.pack(fill=tk.BOTH, expand=True)

# 创建右侧的Frame用于显示违章信息和其他控件
right_frame = tk.Frame(paned_window, width=320, height=720)
right_frame.pack_propagate(False)
paned_window.add(right_frame)

# URL输入字段
url_var = StringVar(root, value="http://127.0.0.1:8888/upload")
url_entry = tk.Entry(right_frame, textvariable=url_var, font=("Arial", 14), width=50)
url_entry.pack(pady=20)

# 上传图像按钮
upload_btn = tk.Button(right_frame, text="上传图片", font=("Arial", 14), command=upload_image)
upload_btn.pack(pady=10)

# 用于显示违章信息的标签
text_label = Label(right_frame, text="违章信息将显示在这里", justify=tk.LEFT, font=("Arial", 14), bg='white',
                   anchor="nw", wraplength=500)
text_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# 运行应用程序
root.mainloop()
