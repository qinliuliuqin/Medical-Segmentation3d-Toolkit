FROM nvidia/cuda:10.0-runtime-ubuntu18.04
WORKDIR /usr/seg

RUN apt-get update -y && apt-get install -y \
 python3.6 \
 python3-pip \
 curl \
 git

RUN pip3 install numpy && \
  pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  
RUN git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
RUN cd Medical-Segmentation3d-Toolkit && pip3 install -e .
