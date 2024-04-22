# 使用官方PyTorch镜像作为基础镜像
FROM pytorch/pytorch:latest

# 设置非交互式安装，避免tzdata等包的配置交互
ARG DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装Flask、Gunicorn、Numpy、Pillow、OpenCV和YOLO
# 使用清华大学镜像源来加速下载
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask gunicorn numpy Pillow opencv-python ultralytics

# 复制所需文件到容器内的工作目录
COPY main_run_linux.py yoloresult.pt se_resnet50_model_85.pth ./

# 暴露端口
EXPOSE 8888

# 当容器启动时使用Gunicorn运行应用
CMD ["gunicorn", "--bind", "0.0.0.0:8888", "main_run_linux:app"]
