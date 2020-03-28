FROM nvidia/cuda:10.0-runtime-ubuntu18.04
WORKDIR /usr/seg

RUN apt-get update -y && apt-get install -y \
 python3.6 \
 python3-pip \
 curl \
 git

RUN pip3 install nano
RUN pip3 install numpy
RUN pip3 install torch torchvision

COPY infer.sh /usr/seg 

RUN git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
RUN git clone https://github.com/qinliuliuqin/Model-Zoo.git
RUN cd Medical-Segmentation3d-Toolkit && pip3 install -e .
