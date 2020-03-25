FROM nvidia/cuda:latest
WORKDIR /usr/seg

# Install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y && apt-get clean
RUN apt-get install -y curl python3.7 python3-pip

RUN git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
RUN cd Medical-Segmentation3d-Toolkit && pip3 install -e .
