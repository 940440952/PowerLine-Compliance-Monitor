# 使用具有CUDA 11.3的官方基础镜像
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# 设置工作目录
WORKDIR /app

# 安装 Python 和必要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt ./

# 安装 Python 依赖项（指定 PyTorch 额外源）
RUN pip3 install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# 复制应用程序代码
COPY main_run_linux.py wsgi.py start_gunicorn.py start_waitress.py ./
COPY ./Font/ ./Font/
COPY ./model/ ./model/

# 暴露端口
EXPOSE 8888




