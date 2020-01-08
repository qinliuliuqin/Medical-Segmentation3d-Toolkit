import numpy as np
import os
import SimpleITK as sitk
import torch


def save_intermediate_results(idxs, crops, masks, outputs, frames, file_names, out_folder):
    """ save intermediate results to training folder

    :param idxs: the indices of crops within batch to save
    :param crops: the batch tensor of image crops
    :param masks: the batch tensor of segmentation crops
    :param outputs: the batch tensor of output label maps
    :param frames: the batch frames
    :param file_names: the batch file names
    :param out_folder: the batch output folder
    :return: None
    """
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)


def crop_image(image, cropping_center, cropping_size, cropping_spacing, interp_method):
    """
    Crop a patch from a volume given the cropping center, cropping size, cropping spacing, and the interpolation method.
    This function DO NOT consider the transformation of coordinate systems, which means the cropped patch has the same
    coordinate system with the given volume.

    :param image: The given volume to be cropped.
    :param cropping_center: The center voxel of the cropped patch in the coordinate system of the given volume.
    :param cropping_size: The size of the cropped patch.
    :param cropping_spacing: The spacing of the cropped patch.
    :param interp_method: The interpolation method, only support 'NN' and 'Linear'.
    :return a cropped patch
    """
    assert isinstance(image, sitk.Image)

    spacing = image.GetSpacing()
    direction = image.GetDirection()

    cropping_center = [int(cropping_center[idx]) for idx in range(3)]
    cropping_size = [int(cropping_size[idx]) for idx in range(3)]
    cropping_spacing = [float(cropping_spacing[idx]) for idx in range(3)]

    cropping_start_point_voxel = [cropping_center[idx] for idx in range(3)]
    for idx in range(3):
        cropping_start_point_voxel[idx] -= int(cropping_size[idx] * cropping_spacing[idx] / spacing[idx]) // 2
    cropping_start_point_world = image.TransformIndexToPhysicalPoint(cropping_start_point_voxel)

    cropping_origin = cropping_start_point_world
    cropping_direction = direction

    transform = sitk.Transform(3, sitk.sitkIdentity)

    if interp_method == 'LINEAR':
        interp_method = sitk.sitkLinear
    elif interp_method == 'NN':
        interp_method = sitk.sitkNearestNeighbor
    else:
        raise ValueError('Unsupported interpolation type.')

    outimage = sitk.Resample(image, cropping_size, transform, interp_method, cropping_origin, cropping_spacing,
                             cropping_direction)

    return outimage


def select_random_voxels_in_multi_class_mask(mask, num_selected, selected_label):
    assert isinstance(mask, sitk.Image)

    return [0]


def set_labels_outside_to_zero(mask, min_label, max_label):

    return mask


def convert_image_to_tensor(image):
    """ Convert an SimpleITK image object to float tensor
    """
    if isinstance(image, sitk.Image):
        tensor = torch.from_numpy(sitk.GetArrayFromImage(image))
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.float()
    elif isinstance(image, list):
        tensor = []
        for i in range(len(image)):
            assert isinstance(image[i], sitk.Image)
            tmp = torch.from_numpy(sitk.GetArrayFromImage(image[i]))
            tmp = torch.unsqueeze(tmp, 0)
            tmp = tmp.float()
            tensor.append(tmp)
            tensor = torch.cat(tensor, 0)

    else:
        raise ValueError('unknown input type')

    return tensor


def convert_tensor_to_image(tensor, dtype):
    pass
    # """ convert tensor to SimpleITK image object """
    # assert isinstance(tensor, torch.Tensor), 'input must be a tensor'
    #
    # data = tensor.cpu().numpy()
    #
    # if tensor.dim() == 3:
    #     # single channel 3d image volume
    #     image = sitk.Image()
    #     if dtype is None:
    #         image.from_numpy(data, dtype=data.dtype)
    #     else:
    #         image.from_numpy(data, dtype=dtype)
    #
    # elif tensor.dim() == 4:
    #     # multi-channel 3d image volume
    #     image = []
    #     for i in range(data.shape[0]):
    #         tmp = Image3d()
    #         if dtype is None:
    #             tmp.from_numpy(data[i], dtype=data.dtype)
    #         else:
    #             tmp.from_numpy(data[i], dtype=dtype)
    #         image.append(tmp)
    #
    # else:
    #     raise ValueError('ToImage() only supports 3-dimsional or 4-dimensional image volume')
    #
    # return image


if __name__ == '__main__':

    image_path = '/home/qinliu/projects/dental/case_67_cbct_patient/org.mha'
    image = sitk.ReadImage(image_path)

    cropping_center = [image.GetSize()[idx] // 2 for idx in range(3)]
    cropping_size = [image.GetSize()[idx] // 2 for idx in range(3)]
    cropping_spacing = [image.GetSpacing()[idx] * 3.0 for idx in range(3)]
    interp_method = 'NN'
    cropped_image = crop_image(image, cropping_center, cropping_size, cropping_spacing, interp_method)

    save_path = '/home/qinliu/cropped_image.mha'
    sitk.WriteImage(cropped_image, save_path, True)