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

1. Inference:
   Print the help info.
   
   Input:
   ```
   $ seg_infer --help
   ```
   Output:
   ```
   usage: seg_infer [-h] [-i INPUT] [-m MODEL] [-o OUTPUT] [-n SEG_NAME]
                 [-g GPU_ID] [--save_image] [--save_prob]

   Inference engine for 3d medical image segmentation It supports multiple kinds
   of input: 1. Single image 2. A text file containing paths of all testing
   images 3. A folder containing all testing images

   optional arguments:
     -h, --help            show this help message and exit
     -i INPUT, --input INPUT
                           input folder/single image/txt file
     -m MODEL, --model MODEL
                           model root folder
     -o OUTPUT, --output OUTPUT
                           output folder for segmentation
     -n SEG_NAME, --seg_name SEG_NAME
                           the name of the segmentation result to be saved
     -g GPU_ID, --gpu_id GPU_ID
                           the gpu id to run model, set to -1 if using cpu only.
     --save_image          whether to save original image
     --save_prob           whether to save all prob maps

   ```
   The following is an example that shows how to get started.
   If you have an image `image.mha`, a segmentation model `model`, and a gpu with devise id `0`, you can run the 
   following command for inference:   
   ```
   $ seg_infer -i ./image.mha -m ./model -o ./result_folder -g 0 
   ```
   WARNING: If you run `$ seg_infer` with no parameters, the program may crash because it will look for the default paths
   which are set according to my environment. 
   
   If you are working on dental project and you want to segment bony structures from CBCT/CT images, you can 
   download pretrained segmentation models on github (the latest model is `model_0429_2020`).
   ```
   $ git clone https://github.com/qinliuliuqin/Model-Zoo/tree/master/Dental/segmentation  
   ```


2. Training:

   Firt, configure the training settings in `config/train_config.py` and then create a training file titled `train.txt`.
   An example train.txt should be the following format:
   ```
   2
   /home/qinliu/train_data/image_1.mha
   /home/qinliu/train_data/label_1.mha
   /home/qinliu/train_data/image_2.mha
   /home/qinliu/train_data/label_2.mha   
   ```
   The first number `2` denotes the pair of traning images and their corresponding labels. Use absolute path.
   You need to set a lot parameters in the configuration file `config/train_config.py`. Please read the comment in the 
   configuration file carefully.
   
   Then, run the following code to start training:
   ```shell
   seg_train -i ./config/train_config.py
   ```

# Requirements
Pytorch=1.3.0
Numpy=1.17.2
SimpleITK=1.2.3
