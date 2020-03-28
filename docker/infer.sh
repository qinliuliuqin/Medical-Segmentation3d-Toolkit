#!/bin/bash

echo '==============================='
echo 'Run Inference'
echo '==============================='

INFER_COMMAND=Medical-Segmentation3d-Toolkit/segmentation3d/seg_infer.py
MODEL_DIR=Model-Zoo/Dental/segmentation/model_0305_2020
DATA_DIR=Model-Zoo/Dental/segmentation/test_data/
OUTPUT_DIR=./result
GPU_ID=0

# Change the call according to your parameters.
python3 ${INFER_COMMAND} -i ${DATA_DIR} -m ${MODEL_DIR} -g ${GPU_ID} -o ${OUTPUT_DIR}
