# 使用官方Python镜像作为基础镜像
FROM python:3.8

# 设置工作目录
WORKDIR /app

# 安装PyTorch和torchvision。注意选择与你的CUDA版本兼容的版本
RUN pip install torch torchvision torchaudio

# 安装Flask和其他依赖库
RUN pip install flask numpy Pillow opencv-python

# 安装ultralytics的YOLOv5。这是从GitHub安装，以确保获取最新版本
RUN pip install ultralytics

# 复制所需文件到容器内的工作目录
COPY main_run_win.py yoloresult.pt resnet5.pth ./

# 暴露端口
EXPOSE 8888

# 当容器启动时运行Python脚本
CMD ["python", "main_run.py"]
