# CUDA 11.7, TensorRT 8.4.1
FROM nvcr.io/nvidia/tensorrt:22.07-py3

ARG DEBIAN_FRONTEND=noninteractive

# Dependencies
RUN apt-get update -y && \
    apt-get install -y sudo git libeigen3-dev wget build-essential cmake  \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libyaml-cpp-dev

# OpenCV
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip && \
    unzip opencv.zip && \
    cd opencv-4.2.0  && \
    mkdir build && cd build && \
    cmake .. && make -j && make install

# Superpoint/Superglue \
RUN git clone https://github.com/changh95/super-mono-vo.git && \
    cd super-mono-vo && \
    mkdir build && cd build && \
    cmake .. && make -j