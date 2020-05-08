import argparse

from segmentation3d.core.seg_infer import segmentation


def main():

    long_description = 'Inference engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Single image\n' \
                       '2. A text file containing paths of all testing images\n'\
                       '3. A folder containing all testing images\n'

    #default_input = '/shenlab/lab_stor6/qinliu/CT_Dental/datasets/test.txt'
    default_input = '/shenlab/lab_stor6/deqiang/Pre_Post_Facial_Data-Ma/original_images'
    default_model = '/shenlab/lab_stor6/qinliu/CT_Dental/models/model_0305_2020/model1_groupnorm_0.4_contrast'
    default_output = '/shenlab/lab_stor6/qinliu/CT_Dental/results/Pre_Post_Facial_Data-Ma_debug'
    default_seg_name = 'seg.mha'
    default_gpu_id =6

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', default=default_input, help='input folder/file for intensity images')
    parser.add_argument('-m', '--model', default=default_model, help='model root folder')
    parser.add_argument('-o', '--output', default=default_output, help='output folder for segmentation')
    parser.add_argument('-n', '--seg_name', default=default_seg_name, help='the name of the segmentation result to be saved')
    parser.add_argument('-g', '--gpu_id', type=int, default=default_gpu_id, help='the gpu id to run model, set to -1 if using cpu only.')
    parser.add_argument('--save_image', help='whether to save original image', action="store_true")
    parser.add_argument('--save_prob', help='whether to save all prob maps', action="store_true")

    args = parser.parse_args()
    segmentation(
        args.input, args.model, args.output, args.seg_name, args.gpu_id, False, True, args.save_image, args.save_prob
    )


if __name__ == '__main__':
    main()
