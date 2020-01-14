# Introduction

3d medical image segmentation engine

# Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit.git
   cd Medical-Segmentation3d-Toolkit/segmentation3d
   ```
2. Configure the training settings in `config/config.py`.
   The format of the training txt file should be:
   ```
   line1: number of traning images, eg. 200
   line2: the absolute path of the first training image, eg. /your-image-folder/image.mha
   line3: the absolute path of the label of the first image, eg. /your-label-folder/label.mha
   ...
   ```
   
3. Train the model:
 
   ```shell
   python seg_train.py
   ```
   
4. Test the model:
   ```shell
   python seg_infer.py
   ```
