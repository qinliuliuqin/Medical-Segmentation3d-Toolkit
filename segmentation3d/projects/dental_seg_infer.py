import os

from segmentation3d.core.seg_infer import segmentation
from segmentation3d.utils.dicom_helper import write_binary_dicom_series


def dental_segmentation(input_folder, model_folder, save_folder, gpu_id):
    """
    This interface is only used for the integration of auto-segmentation function into AA software.
    :param input_folder:        The input dicom folder
    :param model_folder:        The folder contains trained models
    :param gpu_id:              Which gpu to use, by default, 0
    :return: None
    """
    assert os.path.isdir(input_folder) and os.path.isdir(model_folder)

    mask = segmentation(input_folder, model_folder, '', '', gpu_id, True, False, False, False)

    mask_name = os.path.split(input_folder)[-1]
    write_binary_dicom_series(mask, os.path.join(save_folder, '{}_mandible'.format(mask_name)), 1, 100)
    write_binary_dicom_series(mask, os.path.join(save_folder, '{}_midface'.format(mask_name)), 2, 100)
