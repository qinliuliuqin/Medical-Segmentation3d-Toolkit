# Introduction

Medical image 3D segmentation engine.

# Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
   cd Medical-Segmentation3d-Toolkit/segmentation3d
   ```
2. Configure the training settings in `config/train_config.py`.
   The format of the training txt file should be:
   line1: number of traning images, eg. 2
   line2: the absolute path of the first training image, eg. /your-image-folder/image.mha
   line3: the absolute path of the label of the first image, eg. /your-label-folder/label.mha   

   eg.
   ```
   2
   /home/qinliu/train_data/image_1.mha
   /home/qinliu/train_data/label_1.mha
   /home/qinliu/train_data/image_2.mha
   /home/qinliu/train_data/label_2.mha   
   ...
   ```
   
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
