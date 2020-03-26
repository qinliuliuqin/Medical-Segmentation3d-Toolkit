This repository contains a general 3D segmentation engine for mandible and midface segmentation from CT/CBCT image.
As we only tested the code in linux (Ubuntu 18.04) OS, we highly recommend you to run it in the same OS.
In addition to that, please install nvidia-docker first, and then, follow the steps listed below to get your segmentation results.

# Step 1: Download docker image of the repository.

Run command: "sudo docker pull qinliu19/vseg_test:3.0"
It may take a while to download this docker image because of its large size. 
After the image is downloaded, you will see the message "Status: Downloaded newer image for qinliu19/vseg_test:3.0".

Run command: "sudo nvidia-docker run --network=host -it qinliu19/vseg_test:3.0 /bin/bash" to get the shell of the container.
You will see the path: "root@[container-id]:/usr/seg#". Remember this container id because it will be used later.

# Step 2: Run test case in the docker image
Run the test script to check if the docker was correctly downloaded.
   ```shell
   ./infer.sh
   ```   

Run the following command to learn how to use the test function:
   ```shell
   seg_infer --help
   ```   
You will see the following usage instructions:
   ```shell
   usage: seg_infer [-h] [-i INPUT] [-m MODEL] [-o OUTPUT] [-n SEG_NAME]
                 [-g GPU_ID] [--save_image] [--save_prob] [--save_uncertainty]

Inference engine for 3d medical image segmentation It supports multiple kinds
of input: 1. Single image 2. A text file containing paths of all testing
images 3. A folder containing all testing images

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input folder/file for intensity images
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
  --save_uncertainty    whether to save single prob map
   ```   

Below is an example:
   ```shell
   seg_infer -i ./test_data/test.mha -m ./model_0305_2020 -o ./results
   ```

Step 3: Test your own data

First, open a new shell in your host machine, and copy your data to the container by runing: "sudo docker cp [test-folder] [container-id]:/usr/seg/[test-folder]".
The test-folder which contains your test images is copying from your host machine to the container that you are running now. You can check it in your container.
After this step, you can close the new shell in your host machine.

Second, run the command in your container: "vseg_test -i "./usr/seg/[test-folder]". The segmentation results will be saved in the folder "/usr/seg/results". 
You can copy all the test results from the container to your host machine by running: "sudo docker cp [container-id]:/usr/seg/results ."
