FROM nvidia/cuda:10.0-runtime-ubuntu18.04
WORKDIR /usr/seg

RUN apt-get update -y && apt-get install -y \
 python3.6 \
 python3-pip \
 curl \
 git

RUN pip3 install numpy && \
  pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

RUN git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
RUN cd Medical-Segmentation3d-Toolkit && pip3 install -e .
