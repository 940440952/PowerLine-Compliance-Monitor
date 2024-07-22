# 使用官方的 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录的内容到工作目录
COPY main_run_linux.py wsgi.py requirements.txt ./
COPY ./Font/ ./Font/
COPY ./model/ ./model/

# 安装必要的依赖项
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露应用程序的端口
EXPOSE 8888

# 启动命令
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8888", "wsgi:app"]


