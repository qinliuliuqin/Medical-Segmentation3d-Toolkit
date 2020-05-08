from segmentation3d.core.seg_infer import segmentation


def dental_segmentation(input_folder, model_folder, gpu_id):
    """
    This interface is only used for the integration of auto-segmentation function into AA software.
    :param input_folder:        The input dicom folder
    :param model_folder:        The folder contains trained models
    :param gpu_id:              Which gpu to use, by default, 0
    :return: A list containing binary masks of mandible and midface-both are in SimpleITK Image type.
    """
    mask = segmentation(input_folder, model_folder, '', '', gpu_id, True, False, False, False)

    # mask transformation-convert multi-class masks to binary mask with specified label
    binary_masks = []
    # TO BE DONE
       
    return binary_masks
