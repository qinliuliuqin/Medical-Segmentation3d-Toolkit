# Introduction

3D Medical Image Segmentation Engine.

# Installation
This codebase is only tested on Linux (Ubuntu).
   ```shell
   git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
   cd Medical-Segmentation3d-Toolkit
   pip install -e .
   ```
Do not forget the last '.' that indicates the current folder.

# Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
   cd Medical-Segmentation3d-Toolkit/segmentation3d
   ```
2. Configure the training settings in `config/train_config.py` and then create a training file titled `train.txt`.
   An example train.txt can be:
   ```
   2
   /home/qinliu/train_data/image_1.mha
   /home/qinliu/train_data/label_1.mha
   /home/qinliu/train_data/image_2.mha
   /home/qinliu/train_data/label_2.mha   
   ```
   The number `2` in the first line denotes the number of traning images.
   The absolute path of training images and their corresponding labels should be listed in the following lines.
   
3. Train the model:
 
   ```shell
   python seg_train.py
   ```
   
4. Configure the testing settings in `config/infer_config.py`.

5. Test the model:
   ```shell
   python seg_infer.py
   ```
# Requirements
Pytorch=1.1.0
Numpy=1.17.2
SimpleITK=1.2.3
