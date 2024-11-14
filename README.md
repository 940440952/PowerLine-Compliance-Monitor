docker build -t test .
运行docker容器
docker run -p 8888:8888 --gpus all --name testContainer test /bin/bash
python3 start_gunicorn.py
