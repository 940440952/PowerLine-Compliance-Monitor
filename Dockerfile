# 使用官方Python镜像作为基础镜像
FROM python:3.8

# 安装系统依赖项
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
&& rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装PyTorch、torchvision和其他依赖
# 使用清华大学镜像源来加速下载
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio

# 安装Flask、Gunicorn、Numpy、Pillow、OpenCV和YOLO
# 使用清华大学镜像源来加速下载
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask gunicorn numpy Pillow opencv-python ultralytics

# 复制所需文件到容器内的工作目录
COPY main_run_linux.py yoloresult.pt se_resnet50_model_85.pth ./

# 暴露端口
EXPOSE 8888

# 当容器启动时使用Gunicorn运行应用
CMD ["gunicorn", "--bind", "0.0.0.0:8888", "main_run_linux:app"]
