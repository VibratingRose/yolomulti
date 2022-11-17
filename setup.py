from setuptools import find_packages, setup

setup(
    name="yolo",
    version="0.6.2",
    author="muchun",
    author_email="muchun563@163.com",
    description="based on yolov5 r6.2",

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)
