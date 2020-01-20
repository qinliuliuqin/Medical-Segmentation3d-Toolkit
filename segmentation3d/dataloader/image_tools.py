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
    Get the frame of the given image. An image frame contains the origin, spacing, and direction of a image.

    :parma image: a SimpleITK image
    :return frame: the frame packed in a numpy array
    """
    assert isinstance(image, sitk.Image)

    frame = []
    frame.extend(list(image.GetSpacing()))
    frame.extend(list(image.GetOrigin()))
    frame.extend(list(image.GetDirection()))

    return np.array(frame, dtype=np.float32)


def set_image_frame(image, frame):
    """
    Set the frame of the SimpleITK image

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
    """
    Save intermediate results to training folder

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
            cls_num = outputs.size()[1]
            for cls in range(cls_num):
                output = convert_tensor_to_image(outputs[i, cls].data, dtype=np.float32)
                set_image_frame(output, frames[i].numpy())
                sitk.WriteImage(output, os.path.join(case_out_folder, 'batch_{}_output_{}.nii.gz'.format(i, cls)))


def crop_image(image, cropping_center, cropping_size, cropping_spacing, interp_method):
    """
    Crop a patch from a volume given the cropping center, cropping size, cropping spacing, and the interpolation method.
    This function DO NOT consider the transformation of coordinate systems, which means the cropped patch has the same
    coordinate system with the given volume.

    :param image: the given volume to be cropped.
    :param cropping_center: the center of the cropped patch in the world coordinate system of the given volume.
    :param cropping_size: the voxel coordinate size of the cropped patch.
    :param cropping_spacing: the voxel spacing of the cropped patch.
    :param interp_method: the interpolation method, only support 'NN' and 'Linear'.
    :return a cropped patch
    """
    assert isinstance(image, sitk.Image)

    cropping_center = [float(cropping_center[idx]) for idx in range(3)]
    cropping_size = [int(cropping_size[idx]) for idx in range(3)]
    cropping_spacing = [float(cropping_spacing[idx]) for idx in range(3)]

    cropping_physical_size = [cropping_size[idx] * cropping_spacing[idx] for idx in range(3)]
    cropping_start_point_world = [cropping_center[idx] - cropping_physical_size[idx] / 2.0 for idx in range(3)]

    cropping_origin = cropping_start_point_world
    cropping_direction = image.GetDirection()

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


def copy_image(source_image, voi_center, voi_size, target_image, interpolation='NN'):
    """
    Copy data from source image to target image in the volume of interest.

    :param source_image: the source image.
    :param voi_center: the center of the interested volume, unit: mm
    :param voi_size: the size of the interested volume, unit: mm
    :param target_image: the target image
    :param interpolation: the interpolation type, only support 'NN' yet.
    :return None
    """
    assert isinstance(source_image, sitk.Image)
    assert isinstance(target_image, sitk.Image)

    image_size = target_image.GetSize()

    sp_world = [float(voi_center[idx] - voi_size[idx] / 2.0) for idx in range(3)]
    sp_voxel = np.floor(target_image.TransformPhysicalPointToContinuousIndex(sp_world))
    sp_voxel = [int(min(max(0, sp_voxel[idx]), image_size[idx] - 1)) for idx in range(3)]

    ep_world = [float(sp_world[idx] + voi_size[idx]) for idx in range(3)]
    ep_voxel = np.ceil(target_image.TransformPhysicalPointToContinuousIndex(ep_world))
    ep_voxel = [int(min(max(0, ep_voxel[idx]), image_size[idx] - 1)) for idx in range(3)]

    source_size = source_image.GetSize()
    for idx in range(sp_voxel[0], ep_voxel[0] + 1):
        for idy in range(sp_voxel[1], ep_voxel[1] + 1):
            for idz in range(sp_voxel[2], ep_voxel[2] + 1):
                world = target_image.TransformIndexToPhysicalPoint([idx, idy, idz])

                # Nearest Neighbor interpolation
                voxel_in_source = source_image.TransformPhysicalPointToContinuousIndex(world)
                voxel_in_source_nn = [int(round(voxel_in_source[idx])) for idx in range(3)]

                if voxel_in_source_nn[0] >= 0 and voxel_in_source_nn[0] < source_size[0] and \
                    voxel_in_source_nn[1] >= 0 and voxel_in_source_nn[1] < source_size[1] and \
                    voxel_in_source_nn[2] >= 0 and voxel_in_source_nn[2] < source_size[2]:
                    target_image[idx, idy, idz] = source_image.GetPixel(voxel_in_source_nn)


def image_partition_by_fixed_size(image, partition_size):
    """
    Split image by fixed size.

    :param image: the input image to be spilt
    :param partition_size: the physical size of each partition
    :return partition_centers: the list containing center voxel of each partition
    """
    image_size, image_spacing, image_origin = image.GetSize(), image.GetSpacing(), image.GetOrigin()
    image_physical_size = [float(image_size[idx] * image_spacing[idx]) for idx in range(3)]

    partition_number = [int(np.ceil(image_physical_size[idx] / partition_size[idx])) for idx in range(3)]
    partition_centers = []
    for idx in range(0, partition_number[0]):
        for idy in range(0, partition_number[1]):
            for idz in range(0, partition_number[2]):
                center = [float(image_origin[0] + partition_size[0] * (0.5 + idx)),
                          float(image_origin[1] + partition_size[1] * (0.5 + idy)),
                          float(image_origin[2] + partition_size[2] * (0.5 + idz))]

                for index in range(3):
                    center_maximum = image_origin[index] + image_physical_size[index] - partition_size[index] / 2.0
                    center[index] = min(center[index], center_maximum)

                partition_centers.append(center)

    return partition_centers


def normalize_image(image, mean, std, clip, clip_min=-1.0, clip_max=1.0):
    """
    Normalize image by setting mean and standard deviation.
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    image_npy = (image_npy - mean) / std

    if clip:
        image_npy[image_npy < clip_min] = clip_min
        image_npy[image_npy > clip_max] = clip_max

    normalized_image = sitk.GetImageFromArray(image_npy)
    set_image_frame(normalized_image, get_image_frame(image))
    normalized_image = sitk.Cast(normalized_image, image.GetPixelID())

    return normalized_image


def percentiles(image, percentiles):
    """
    Get image percentile
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    results = np.percentile(image_npy, percentiles)
    return results


def select_random_voxels_in_multi_class_mask(mask, num_selected, selected_label):
    """
    Randomly select a list of voxels with the given label in the mask

    :param mask: A multi-class label image
    :param num_selected: the number of voxels to be selected
    :param selected_label: the label to which the selected voxels belong
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
    """
    Convert an SimpleITK image object to float tensor
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
    """
    convert tensor to SimpleITK image object
    """
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