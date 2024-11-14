import subprocess


def start_gunicorn():
    command = [
        'gunicorn',
        '-w', '4',  # 使用4个worker
        '-b', '0.0.0.0:8888',  # 绑定到所有IP地址的8888端口
        'wsgi:app'  # 指定wsgi模块和应用对象
    ]

    try:
        # 通过subprocess启动gunicorn
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Gunicorn failed to start: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    start_gunicorn()

