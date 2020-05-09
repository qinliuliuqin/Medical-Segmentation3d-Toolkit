import argparse
import os

from segmentation3d.core.seg_infer import segmentation
from segmentation3d.utils.dicom_helper import write_binary_dicom_series


def dental_segmentation(input_dicom_folder, model_folder, save_dicom_folder, gpu_id):
    """
    This interface is only used for the integration of auto-segmentation function into AA software.
    :param input_dicom_folder:        The input dicom folder
    :param model_folder:              The folder contains trained models
    :param save_dicom_folder:         The folder to save binary masks of mandible and midfacec in dicom format
    :param gpu_id:                    Which gpu to use, by default, 0
    :return: None
    """
    assert os.path.isdir(input_dicom_folder) and os.path.isdir(model_folder)

    mask = segmentation(input_dicom_folder, model_folder, '', '', gpu_id, True, False, False, False)
    mask_name = os.path.split(input_dicom_folder)[-1]
    write_binary_dicom_series(mask[0], os.path.join(save_dicom_folder, '{}_midface'.format(mask_name)), 1, 100)
    write_binary_dicom_series(mask[0], os.path.join(save_dicom_folder, '{}_mandible'.format(mask_name)), 2, 100)


def main():

    long_description = 'Inference interface for dental segmentation.'

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', help='input dicom folder')
    parser.add_argument('-m', '--model', help='model folder')
    parser.add_argument('-o', '--output', help='output dicom folder to save binary masks.')
    parser.add_argument('-g', '--gpu_id', type=int, help='the gpu id to run model, set to -1 if using cpu only.')

    args = parser.parse_args()
    dental_segmentation(args.input, args.model, args.output, args.gpu_id)


if __name__ == '__main__':
    main()
