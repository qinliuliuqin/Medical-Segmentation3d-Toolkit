from __future__ import print_function
import argparse
import os
import time
import numpy as np


def segmentation(input_path, model_folder, output_folder,seg_name='seg.mha', gpu_id=0, save_image=True,
                 save_single_prob=False):
    """ volumetric image segmentation engine
    :param input_path:          a path of text file, a single image file
                                or a root dir with all image files
    :param model_folder:        path of trained model
    :param output_folder:       path of out folder
    :param gpu_id:              which gpu to use, by default, 0
    :param save_image           whether to save original image
    :return: None
    """
    total_test_time = 0


def main():

    long_description = 'Training engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Image list txt file\n' \
                       '2. Single image file\n' \
                       '3. A folder that contains all testing images\n'
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input', type=str, help='input folder/file for intensity images')
    parser.add_argument('-m', '--model', type=str, help='model root folder')
    parser.add_argument('-o', '--output', type=str, help='output folder for segmentation')
    parser.add_argument('-n', '--seg_name', default='seg.mha', help='the name of the segmentation result to be saved')
    parser.add_argument('-g', '--gpu_id', default='0', help='the gpu id to run model')
    parser.add_argument('--save_image', help='whether to save original image', action="store_true")
    parser.add_argument('--save_single_prob', help='whether to save single prob map', action="store_true")
    args = parser.parse_args()

    segmentation(args.input, args.model, args.output, args.seg_name, int(args.gpu_id), args.save_image,
                 args.save_single_prob)


if __name__ == '__main__':
    main()