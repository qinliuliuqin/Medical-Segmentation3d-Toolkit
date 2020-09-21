import argparse

from segmentation3d.core.seg_train import train


def main():

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    long_description = "Training engine for 3d medical image segmentation"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='/home/qinliu19/projects/Medical-Segmentation3d-Toolkit/segmentation3d/config/train_config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()
