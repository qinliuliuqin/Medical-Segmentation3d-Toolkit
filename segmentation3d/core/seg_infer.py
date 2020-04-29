import argparse
import glob
import importlib
import torch.nn as nn
import os
import SimpleITK as sitk
import time
import torch
import numpy as np
from easydict import EasyDict as edict

from segmentation3d.utils.file_io import load_config, readlines
from segmentation3d.utils.model_io import get_checkpoint_folder
from segmentation3d.dataloader.image_tools import resample, convert_image_to_tensor, convert_tensor_to_image, \
    copy_image, image_partition_by_fixed_size, resample_spacing, add_image_value, pick_largest_connected_component, \
    remove_small_connected_component
from segmentation3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer
from segmentation3d.vis.vtk_rendering import vtk_surface_rendering, get_color_dict


def read_test_txt(txt_file):
    """ read single-modality txt file
    :param txt_file: image list txt file path
    :return: a list of image path list, list of image case names
    """
    lines = readlines(txt_file)
    case_num = int(lines[0])

    if len(lines) - 1 != case_num:
        raise ValueError('case num do not equal path num!')

    file_name_list, file_path_list = [], []
    for i in range(case_num):
        im_msg = lines[1 + i]
        im_msg = im_msg.strip().split()
        im_name = im_msg[0]
        im_path = im_msg[1]
        if not os.path.isfile(im_path):
            raise ValueError('image not exist: {}'.format(im_path))
        file_name_list.append(im_name)
        file_path_list.append(im_path)

    return file_name_list, file_path_list


def read_test_folder(folder_path):
    """ read single-modality input folder
    :param folder_path: image file folder path
    :return: a list of image path list, list of image case names
    """
    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    file = []
    for suf in suffix:
        file += glob.glob(os.path.join(folder_path, '*' + suf))

    file_name_list, file_path_list = [], []
    for im_pth in sorted(file):
        _, im_name = os.path.split(im_pth)
        for suf in suffix:
            idx = im_name.find(suf)
            if idx != -1:
                im_name = im_name[:idx]
                break
        file_name_list.append(im_name)
        file_path_list.append(im_pth)

    return file_name_list, file_path_list


def load_seg_model(model_folder, gpu_id=0):
    """ load segmentation model from folder
    :param model_folder:    the folder containing the segmentation model
    :param gpu_id:          the gpu device id to run the segmentation model
    :return: a dictionary containing the model and inference parameters
    """
    assert os.path.isdir(model_folder), 'Model folder does not exist: {}'.format(model_folder)

    # load inference config file
    latest_checkpoint_dir = get_checkpoint_folder(os.path.join(model_folder, 'checkpoints'), -1)
    infer_cfg = load_config(os.path.join(latest_checkpoint_dir, 'infer_config.py'))
    model = edict()
    model.infer_cfg = infer_cfg

    # load model state
    chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')

    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(int(gpu_id))
        # load network module
        state = torch.load(chk_file)
        net_module = importlib.import_module('segmentation3d.network.' + state['net'])
        net = net_module.SegmentationNet(state['in_channels'], state['out_channels'])
        net = nn.parallel.DataParallel(net, device_ids=[0])
        net.load_state_dict(state['state_dict'])
        net.eval()
        net = net.cuda()
        del os.environ['CUDA_VISIBLE_DEVICES']

    else:
        state = torch.load(chk_file, map_location='cpu')
        net_module = importlib.import_module('segmentation3d.network.' + state['net'])
        net = net_module.SegmentationNet(state['in_channels'], state['out_channels'])
        net.load_state_dict(state['state_dict'])
        net.eval()

    model.net = net
    model.spacing, model.max_stride, model.interpolation = state['spacing'], state['max_stride'], state['interpolation']
    model.in_channels, model.out_channels = state['in_channels'], state['out_channels']

    model.crop_normalizers = []
    for crop_normalizer in state['crop_normalizers']:
        if crop_normalizer['type'] == 0:
            mean, stddev, clip = crop_normalizer['mean'], crop_normalizer['stddev'], crop_normalizer['clip']
            model.crop_normalizers.append(FixedNormalizer(mean, stddev, clip))

        elif crop_normalizer['type'] == 1:
            clip_sigma = crop_normalizer['clip_sigma']
            model.crop_normalizers.append(AdaptiveNormalizer(clip_sigma))

        else:
            raise ValueError('Unsupported normalization type.')

    return model


def segmentation_voi(model, iso_image, start_voxel, end_voxel, use_gpu):
    """ Segment the volume of interest
    :param model:           the loaded segmentation model.
    :param iso_image:       the image volume that has the same spacing with the model's resampling spacing.
    :param start_voxel:     the start voxel of the volume of interest (inclusive).
    :param end_voxel:       the end voxel of the volume of interest (exclusive).
    :param use_gpu:         whether to use gpu or not, bool type.
    :return:
      mean_prob_maps:        the mean probability maps of all classes
      std_maps:              the standard deviation maps of all classes
    """
    assert isinstance(iso_image, sitk.Image)

    roi_image = iso_image[start_voxel[0]:end_voxel[0], start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2]]

    if model['crop_normalizers'] is not None:
        roi_image = model.crop_normalizers[0](roi_image)

    roi_image_tensor = convert_image_to_tensor(roi_image).unsqueeze(0)
    if use_gpu:
        roi_image_tensor = roi_image_tensor.cuda()

    bayesian_iteration = model['infer_cfg'].general.bayesian_iteration
    with torch.no_grad():
        probs = model['net'](roi_image_tensor)
        probs = torch.unsqueeze(probs, 0)
        for i in range(bayesian_iteration - 1):
            probs = torch.cat((probs, torch.unsqueeze(model['net'](roi_image_tensor), 0)), 0)
        mean_probs, stddev_maps = torch.mean(probs, 0), torch.std(probs, 0)

    num_classes = model['out_channels']
    assert num_classes == mean_probs.shape[1]

    # return the average probability map
    mean_prob_maps, std_maps = [], []
    for idx in range(num_classes):
        mean_prob = convert_tensor_to_image(mean_probs[0][idx].data, dtype=np.float)
        mean_prob.CopyInformation(roi_image)
        mean_prob_maps.append(mean_prob)

        std_map = convert_tensor_to_image(stddev_maps[0][idx].data, dtype=np.float)
        std_map.CopyInformation(roi_image)
        std_maps.append(std_map)

    return mean_prob_maps, std_maps


def segmentation(input_path, model_folder, output_folder, seg_name, gpu_id, save_image, save_prob, save_uncertainty):
    """ volumetric image segmentation engine
    :param input_path:          The path of text file, a single image file or a root dir with all image files
    :param model_folder:        The path of trained model
    :param output_folder:       The path of out folder
    :param gpu_id:              Which gpu to use, by default, 0
    :param save_image:          Whether to save original image
    :param save_prob:           Whether to save all probability maps
    :param save_uncertainty:    Whether to save all uncertainty maps
    :return: None
    """

    # load model
    begin = time.time()
    model = load_seg_model(model_folder, gpu_id)
    load_model_time = time.time() - begin

    # load test images
    if os.path.isfile(input_path):
        if input_path.endswith('.txt'):
            file_name_list, file_path_list = read_test_txt(input_path)
        else:
            if input_path.endswith('.mhd') or input_path.endswith('.mha') or input_path.endswith('.nii.gz') or \
                    input_path.endswith('.nii') or input_path.endswith('.hdr') or input_path.endswith('.image3d'):
                im_name = os.path.basename(input_path)
                file_name_list = [im_name]
                file_path_list = [input_path]

            else:
                raise ValueError('Unsupported input path.')

    elif os.path.isdir(input_path):
        file_name_list, file_path_list = read_test_folder(input_path)

    else:
        raise ValueError('Unsupported input path.')

    # test each case
    num_success_case = 0
    total_inference_time = 0
    for i, file_path in enumerate(file_path_list):
        print('{}: {}'.format(i, file_path))

        # load image
        begin = time.time()
        image = sitk.ReadImage(file_path, sitk.sitkFloat32)
        read_image_time = time.time() - begin

        iso_image = resample_spacing(image, model['spacing'], model['max_stride'], model['interpolation'])

        num_classes = model['out_channels']
        iso_mean_probs, iso_std_maps = [], []
        for idx in range(num_classes):
            iso_mean_prob = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
            iso_mean_prob.CopyInformation(iso_image)
            iso_mean_probs.append(iso_mean_prob)

            iso_std_map = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
            iso_std_map.CopyInformation(iso_image)
            iso_std_maps.append(iso_std_map)

        partition_type = model['infer_cfg'].general.partition_type
        partition_stride = model['infer_cfg'].general.partition_stride
        if partition_type == 'DISABLE':
            start_voxels = [[0, 0, 0]]
            end_voxels = [[int(iso_image.GetSize()[idx]) for idx in range(3)]]

        elif partition_type == 'SIZE':
            partition_size = model['infer_cfg'].general.partition_size
            max_stride = model['max_stride']
            start_voxels, end_voxels = \
                image_partition_by_fixed_size(iso_image, partition_size, partition_stride, max_stride)

        else:
            raise ValueError('Unsupported partition type!')

        begin = time.time()
        iso_partition_overlap_count = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
        iso_partition_overlap_count.CopyInformation(iso_image)
        for idx in range(len(start_voxels)):
            start_voxel, end_voxel = start_voxels[idx], end_voxels[idx]

            voi_mean_probs, voi_std_maps = segmentation_voi(model, iso_image, start_voxel, end_voxel, gpu_id > 0)
            for idy in range(num_classes):
                iso_mean_probs[idy] = copy_image(voi_mean_probs[idy], start_voxel, end_voxel, iso_mean_probs[idy])
                if save_uncertainty:
                    iso_std_maps[idy] = copy_image(voi_std_maps[idy], start_voxel, end_voxel, iso_std_maps[idy])

            iso_partition_overlap_count = add_image_value(iso_partition_overlap_count, start_voxel, end_voxel, 1.0)
            print('{:0.2f}%'.format((idx + 1) / len(start_voxels) * 100))

        iso_partition_overlap_count = sitk.Cast(1.0 / iso_partition_overlap_count, sitk.sitkFloat32)
        for idx in range(num_classes):
            iso_mean_probs[idx] = iso_mean_probs[idx] * iso_partition_overlap_count
            if save_uncertainty:
                iso_std_maps[idx] = iso_std_maps[idx][:] * iso_partition_overlap_count[:]

        # resample to the original spacing
        mean_probs, std_maps = [], []
        for idx in range(num_classes):
            mean_probs.append(resample(iso_mean_probs[idx], image, 'LINEAR'))
            if save_uncertainty:
                std_maps.append(resample(iso_std_maps[idx], image, 'LINEAR'))

        # get segmentation mask from the mean_probability maps
        mean_probs_tensor = convert_image_to_tensor(mean_probs)
        _, mask = mean_probs_tensor.max(0)
        mask = convert_tensor_to_image(mask, dtype=np.int8)
        mask.CopyInformation(image)
        inference_time = time.time() - begin

        begin = time.time()
        case_name = file_name_list[i]
        if not os.path.isdir(os.path.join(output_folder, case_name)):
            os.makedirs(os.path.join(output_folder, case_name))

        # pick the largest component
        if model['infer_cfg'].general.pick_largest_cc:
            mask = pick_largest_connected_component(mask, list(range(1, num_classes)))

        # remove small connected component
        if model['infer_cfg'].general.remove_small_cc > 0:
            threshold = model['infer_cfg'].general.remove_small_cc
            mask = remove_small_connected_component(mask, list(range(1, num_classes)), threshold)

        post_processing_time = time.time() - begin

        begin = time.time()
        # save results
        sitk.WriteImage(mask, os.path.join(output_folder, case_name, seg_name), True)

        if save_image:
            sitk.WriteImage(image, os.path.join(output_folder, case_name, 'org.mha'), True)

        if save_prob:
            for idx in range(num_classes):
                mean_prob_save_path = os.path.join(output_folder, case_name, 'mean_prob_{}.mha'.format(idx))
                sitk.WriteImage(mean_probs[idx], mean_prob_save_path, True)

        if save_uncertainty:
            for idx in range(num_classes):
                std_map_save_path = os.path.join(output_folder, case_name, 'std_map_{}.mha'.format(idx))
                sitk.WriteImage(std_maps[idx], std_map_save_path, True)
        save_time = time.time() - begin

        total_test_time = load_model_time + read_image_time + inference_time + post_processing_time + save_time
        total_inference_time += inference_time
        num_success_case += 1

        print('total test time: {:.2f}, average inference time: {:.2f}'.format(total_test_time,
                                                                               total_inference_time / num_success_case))