# 使用说明

## 1) 基于 Docker 的 Web 模型推理服务
### 构建 Docker 镜像
```bash
docker build -t test .
```
### 可使用cp命令进行修改 Docker 容器，然后使用commit保存容器为新的Docker镜像
```bash
docker cp F:/main_run_linux.py testContainer:./app
docker commit testContainer test:v1
```
### 保存Docker镜像到本地
```bash
docker save -o F:/docker_image/aidetect.tar test:v1
```
### 载入 Docker 镜像
```bash
docker load -i model_detect.tar
```
### 指定镜像运行 Docker 容器
```bash
docker run -it -p 8888:8888 --gpus all --name testContainer test:v1 /bin/bash
```
testContainer为容器名,test:v1为镜像名.

### 启动 Gunicorn 服务
```bash
python3 start_gunicorn.py
```

访问服务后即可进行模型推理。

------------------------------------------

## 2) 使用 PyCharm 或其他 IDE 的本地开发测试

### 运行本地推理脚本
在项目根目录下运行以下命令：
```bash
python run.py
```

通过脚本加载模型并进行本地图片推理，支持输入单张图片或批量处理文件夹中的图片。

---

## 目录说明
项目当前目录应包含`test_images` 与`output_images`文件夹: 
- **test_images**: 用于存放需要推理的输入图片。
- **output_images**: 用于保存推理完成后的结果图片。

确保输入图片存放在 `test_images` 文件夹中，推理结果将自动保存到 `output_images` 文件夹中，`app-run.log`为模型推理日志文件。
