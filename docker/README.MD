This repository contains a general 3D segmentation engine for mandible and midface segmentation from CT/CBCT image.
As we only tested the code in linux (Ubuntu 18.04) OS, we highly recommend you to run it in the same OS.
In addition to that, please install nvidia-docker first, and then, follow the steps listed below to get your segmentation results.

Step 1: Download docker image of the repository.

Run command: "sudo docker pull qinliu19/vseg_test:3.0"
It may take a while to download this docker image because of its large size. 
After the image is downloaded, you will see the message "Status: Downloaded newer image for qinliu19/vseg_test:3.0".

Run command: "sudo nvidia-docker run -it qinliu19/vseg_test:3.0 /bin/bash" to get the shell of the container.
You will see the path: "root@[container-id]:/usr/seg#". Remember this container id because it will be used later.

Step 2: Run test case in the docker image

Run the following command to learn how to use the test function:
   ```shell
   seg_infer --help
   ```   
You will see 

By default, it will segment the test image "/usr/seg/test_data/org.mha", and save the segmentation result to the folder "/usr/seg/results/org".
The ground truth of the test image is "/usr/seg/test_data/gt.mha". 

You can use the following instructions to control the segmentation process:
Usage: vseg_test \
       -i The input folder of test images, or the name of test image. Type=string, default="./test_data/org.mha". \
       -m The model folder. Type=string, default="./model_1130_2019". \
       -o The output folder of test image(s). Type=string, default="./results".  \
       -n The name of the segmentation result to be saved. Type=string, default="seg.mha". \
       -g The gpu id to run model. Type=string, default='0'.

The command can be:
   ```shell
   vseg_test -i ./test_data/test.mha -m ./model_1130_2019 -o ./results
   ```


Step 3: Test your own data

First, open a new shell in your host machine, and copy your data to the container by runing: "sudo docker cp [test-folder] [container-id]:/usr/seg/[test-folder]".
The test-folder which contains your test images is copying from your host machine to the container that you are running now. You can check it in your container.
After this step, you can close the new shell in your host machine.

Second, run the command in your container: "vseg_test -i "./usr/seg/[test-folder]". The segmentation results will be saved in the folder "/usr/seg/results". 
You can copy all the test results from the container to your host machine by running: "sudo docker cp [container-id]:/usr/seg/results ."

Step 4: Train your own model

1. install 'md' and 'md_segmentation3d' in your host machine

   a. Run the following code to copy the two packages from the container to your host machine and then install them.
      ```shell
      cd /your-folder
      sudo docker cp [container-id]:/usr/seg/md-0.1-py3-none-any.whl .
      sudo docker cp [container-id]:/usr/seg/md_segmentation3d-3.5-py3-none-any.whl .
      pip install ./md-0.1-py3-none-any.whl
      pip install ./md_segmentation3d-3.5-py3-none-any.whl
      ```
   b. Run the following code to make sure the packages have been successfully installed.
      ```shell
      vseg_train --help
      ```
      If you see the follwoing information, the two packages have been successfully installed.
      ```shell
      usage: vseg_train [-h] [-i [INPUT]]
        optional arguments:
              -h, --help                    show this help message and exit
              -i [INPUT], --input [INPUT]   volumetric segmentation3d train config file  
      ```

2. clone the project 'vseg_docker' and get into the folder.
   ```shell
   git clone https://github.com/qinliuliuqin/vseg_docker.git
   cd /your-folder/vseg_docker
   ```

3. modify the correct path in the follwoing files: 'config_coarse.py', 'config_fine.py', 'train.txt'

   a. modify the configuration files to be like this:
   ```shell
   # image-segmentation pair list
   # 1) single-modality image training, use txt annotation file
   # 2) multi-modality image training, use csv annotation file
   __C.general.imseg_list = '/your-folder/vseg_docker/model_mmdd_yyyy/train.txt'

   # the output of training models and logs
   __C.general.save_dir = '/your-folder/vseg_docker/model_mmdd_yyyy/coarse'
   ```
   b. modify the training path file to be like this:
   ```shell
   1
   /your-folder/vseg_docker/data/org.mha
   /your-folder/vseg_docker/data/gt.mha
   ```

4. train the coarse and fine models seperately on the given test data.
   ```shell
   vseg_train -i /your-folder/vseg_docker/model_mmdd_yyyy/config_coarse.py
   vseg_train -i /your-folder/vseg_docker/model_mmdd_yyyy/config_fine.py   
   ```
   
5. train the coarse and fine models seperately on a your own data.

   a. copy your training data to the folder '/your-folder/vseg_docker/data' 
   
   b. update the 'train.txt'.
   
   c. run the training code above.






