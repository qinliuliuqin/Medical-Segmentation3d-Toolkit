import numpy as np
import os
import SimpleITK as sitk
import torch


type_conversion_from_numpy_to_sitk = {
    np.int8:     sitk.sitkInt8,
    np.int16:    sitk.sitkInt16,
    np.int32:    sitk.sitkInt32,
    np.int:      sitk.sitkInt32,
    np.int64:    sitk.sitkInt64,
    np.uint8:    sitk.sitkUInt8,
    np.uint16:   sitk.sitkUInt16,
    np.uint32:   sitk.sitkUInt32,
    np.uint64:   sitk.sitkUInt64,
    np.uint:     sitk.sitkUInt32,
    np.float32:  sitk.sitkFloat32,
    np.float64:  sitk.sitkFloat64,
    np.float:    sitk.sitkFloat32
}


def get_image_frame(image):
    """
    Get the frame of the given image. The image frame contains the origin, spacing, and direction of a image.

    :parma image: A SimpleITK image
    :return frame: The frame packed in a numpy array
    """
    assert isinstance(image, sitk.Image)

    frame = []
    frame.extend(list(image.GetSpacing()))
    frame.extend(list(image.GetOrigin()))
    frame.extend(list(image.GetDirection()))

    return np.array(frame, dtype=np.float32)


def set_image_frame(image, frame):
    """ Set the frame of the SimpleITK image

    :param image: the a new frame to the input image.
    :param frame: the new frame of the image. It is a numpy array with 15 elements, with the first three elements
                  representing the spacing, the next three elements representing the origin, and the rest representing
                  the direction.
    """
    assert isinstance(image, sitk.Image)

    spacing = frame[:3].astype(np.double)
    origin = frame[3:6].astype(np.double)
    direction = frame[6:15].astype(np.double)

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)


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

    for i in idxs:

        case_out_folder = os.path.join(out_folder, file_names[i])
        if not os.path.isdir(case_out_folder):
            os.makedirs(case_out_folder)

        if crops is not None:
            images = convert_tensor_to_image(crops[i], dtype=np.float32)
            frame = frames[i].numpy()
            for modality_idx, image in enumerate(images):
                set_image_frame(image, frame)
                sitk.WriteImage(image, os.path.join(case_out_folder, 'batch_{}_crop_{}.nii.gz'.format(i, modality_idx)))

        if masks is not None:
            mask = convert_tensor_to_image(masks[i, 0], dtype=np.int8)
            set_image_frame(mask, frames[i].numpy())
            sitk.WriteImage(mask, os.path.join(case_out_folder, 'batch_{}_mask.nii.gz'.format(i)))

        if outputs is not None:
            output = convert_tensor_to_image(outputs[i, 0].data, dtype=np.float32)
            set_image_frame(output, frames[i].numpy())
            sitk.WriteImage(output, os.path.join(case_out_folder, 'batch_{}_output.nii.gz'.format(i)))


def crop_image(image, cropping_center, cropping_size, cropping_spacing, interp_method):
    """
    Crop a patch from a volume given the cropping center, cropping size, cropping spacing, and the interpolation method.
    This function DO NOT consider the transformation of coordinate systems, which means the cropped patch has the same
    coordinate system with the given volume.

    :param image: The given volume to be cropped.
    :param cropping_center: The center of the cropped patch in the world coordinate system of the given volume.
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

    cropping_start_point_voxel = list(image.TransformPhysicalPointToIndex(cropping_center))
    for idx in range(3):
        cropping_start_point_voxel[idx] -= int(cropping_size[idx] * cropping_spacing[idx] / spacing[idx]) // 2
    cropping_start_point_world = image.TransformIndexToPhysicalPoint(cropping_start_point_voxel)

    cropping_origin = cropping_start_point_world
    cropping_direction = direction

    if interp_method == 'LINEAR':
        interp_method = sitk.sitkLinear
    elif interp_method == 'NN':
        interp_method = sitk.sitkNearestNeighbor
    else:
        raise ValueError('Unsupported interpolation type.')

    transform = sitk.Transform(3, sitk.sitkIdentity)
    outimage = sitk.Resample(image, cropping_size, transform, interp_method, cropping_origin, cropping_spacing,
                             cropping_direction)

    return outimage


def select_random_voxels_in_multi_class_mask(mask, num_selected, selected_label):
    """ Randomly select a list of voxels with the given label in the mask

    :param mask: A multi-class label image
    :param num_selected: The number of voxels to be selected
    :param selected_label: The label to which the selected voxels belong
    """
    assert isinstance(mask, sitk.Image)

    mask_npy = sitk.GetArrayFromImage(mask)
    valid_voxels = np.argwhere(mask_npy == selected_label)

    selected_voxels = []
    while len(valid_voxels) > 0 and len(selected_voxels) < num_selected:
        selected_index = np.random.randint(0, len(valid_voxels))
        selected_voxel = valid_voxels[selected_index]
        selected_voxels.append(selected_voxel[::-1])

    return selected_voxels


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
    """ convert tensor to SimpleITK image object """
    assert isinstance(tensor, torch.Tensor), 'input must be a tensor'

    data = tensor.cpu().numpy()

    if tensor.dim() == 3:
        # single channel 3d image volume
        image = sitk.GetImageFromArray(data)

        if dtype is not None and dtype in type_conversion_from_numpy_to_sitk.keys():
            sitk_type = type_conversion_from_numpy_to_sitk[dtype]
            image = sitk.Cast(image, sitk_type)

    elif tensor.dim() == 4:
        # multi-channel 3d image volume
        image = []
        for i in range(data.shape[0]):
            tmp = sitk.GetImageFromArray(data[i])

            if dtype is not None and dtype in type_conversion_from_numpy_to_sitk.keys():
                sitk_type = type_conversion_from_numpy_to_sitk[dtype]
                tmp = sitk.Cast(tmp, sitk_type)
            image.append(tmp)
    else:
        raise ValueError('Only supports 3-dimsional or 4-dimensional image volume')

    return image


def test_crop_image():
    image_path = '/home/qinliu/projects/dental/case_67_cbct_patient/org.mha'
    image = sitk.ReadImage(image_path)

    cropping_center = [image.GetSize()[idx] // 2 for idx in range(3)]
    cropping_size = [image.GetSize()[idx] // 2 for idx in range(3)]
    cropping_spacing = [image.GetSpacing()[idx] * 3.0 for idx in range(3)]
    interp_method = 'NN'
    cropped_image = crop_image(image, cropping_center, cropping_size, cropping_spacing, interp_method)

    save_path = '/home/qinliu/cropped_image.mha'
    sitk.WriteImage(cropped_image, save_path, True)


def test_convert_tensor_to_image():
    tensor = torch.randn(128, 128, 128)
    image = convert_tensor_to_image(tensor, dtype=np.int)

    save_path = '/home/qinliu/random_image.mha'
    sitk.WriteImage(image, save_path, True)


if __name__ == '__main__':

    test_convert_tensor_to_image()