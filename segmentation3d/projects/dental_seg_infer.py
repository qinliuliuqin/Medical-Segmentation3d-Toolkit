from segmentation3d.core.seg_infer import segmentation


def dental_segmentation(input_folder, model_folder, gpu_id):
    """
    This interface is only used for the integration of auto-segmentation into AA software.
    :param input_path:          The path of text file, a single image file or a root dir with all image files
    :param model_folder:        The path of trained model
    :param output_folder:       The path of out folder
    :param gpu_id:              Which gpu to use, by default, 0
    :param return_mask:         Whether to return mask
    :param save_mask:           Whether to save mask
    :return: None
    """
    mask = segmentation(input_folder, model_folder, '', '', gpu_id, True, False, False, False)

    # mask transformation-convert multi-class masks to binary mask with specified label
    # TO BE DONE