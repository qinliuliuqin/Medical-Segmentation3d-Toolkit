import argparse
import glob
import importlib
import os
import SimpleITK as sitk
import time
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict as edict

from segmentation3d.utils.file_io import load_config, readlines
from segmentation3d.utils.model_io import get_checkpoint_folder
from segmentation3d.dataloader.image_tools import get_image_frame, set_image_frame, crop_image, resample, \
  convert_image_to_tensor, convert_tensor_to_image, copy_image, image_partition_by_fixed_size, resample_spacing
from segmentation3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer


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

  model = edict()

  # load inference config file
  latest_checkpoint_dir = get_checkpoint_folder(os.path.join(model_folder, 'checkpoints'), -1)
  infer_cfg = load_config(os.path.join(latest_checkpoint_dir, 'infer_config.py'))
  model.infer_cfg = infer_cfg

  # load model state
  chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')

  if gpu_id >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(int(gpu_id))

    # load network module
    state = torch.load(chk_file)
    net_module = importlib.import_module('segmentation3d.network.' + state['net'])
    net = net_module.SegmentationNet(state['in_channels'], state['out_channels'], state['dropout'])
    net = nn.parallel.DataParallel(net)
    net.load_state_dict(state['state_dict'])
    net.eval()
    net = net.cuda()

    del os.environ['CUDA_VISIBLE_DEVICES']

  else:
    state = torch.load(chk_file, map_location='cpu')
    net_module = importlib.import_module('segmentation3d.network.' + state['net'])
    net = net_module.SegmentationNet(state['in_channels'], state['out_channels'], state['dropout'])
    net.load_state_dict(state['state_dict'])
    net.eval()

  model.net = net
  model.spacing, model.max_stride, model.interpolation = state['spacing'], state['max_stride'], state['interpolation']

  model.crop_normalizers = []
  for crop_normalizer in state['crop_normalizers']:
    if crop_normalizer['type'] == 0:
      mean, stddev, clip = crop_normalizer['mean'], crop_normalizer['stddev'], crop_normalizer['clip']
      model.crop_normalizers.append(FixedNormalizer(mean, stddev, clip))

    elif crop_normalizer['type'] == 1:
      min_p, max_p, clip = crop_normalizer['min_p'], crop_normalizer['max_p'], crop_normalizer['clip']
      model.crop_normalizers.append(AdaptiveNormalizer(min_p, max_p, clip))

    else:
      raise ValueError('Unsupported normalization type.')

  return model


def segmentation_voi(model, iso_image, start_voxel, end_voxel, use_gpu, save_prob_index):
  """ Segment the volume of interest
  :param model:           the loaded segmentation model.
  :param iso_image:       the image volume that has the same spacing with the model's resampling spacing.
  :param start_voxel:     the start voxel of the volume of interest (inclusive).
  :param end_voxel:       the end voxel of the volume of interest (exclusive).
  :param use_gpu:         whether to use gpu or not, bool type.
  :param save_prob_index: the index of class to save its mean probability map
  :return:
    mask:                 the segmentation mask
    mean_prob:            the mean probability map of a given class
    std_map:              the standard deviation map of the mean probability map of the given class
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
    mean_probs, stddev_probs = torch.mean(probs, 0), torch.std(probs, 0)

  # return segmentation mask
  _, mask = mean_probs.max(1)
  mask = convert_tensor_to_image(mask[0].data, dtype=np.int8)
  mask.CopyInformation(roi_image)

  # return the average probability map
  mean_prob = None
  if save_prob_index >= 0:
    mean_prob = convert_tensor_to_image(mean_probs[0][save_prob_index].data, dtype=np.float)
    mean_prob.CopyInformation(roi_image)

  # return the standard deviation of probability map
  std_map = None
  if save_prob_index >= 0:
    std_map = convert_tensor_to_image(stddev_probs[0][save_prob_index].data, dtype=np.float)
    std_map.CopyInformation(roi_image)

  return mask, mean_prob, std_map


def segmentation(input_path, model_folder, output_folder, seg_name, gpu_id, save_image, save_prob_index):
    """ volumetric image segmentation engine
    :param input_path:          a path of text file, a single image file
                                or a root dir with all image files
    :param model_folder:        path of trained model
    :param output_folder:       path of out folder
    :param gpu_id:              which gpu to use, by default, 0
    :param save_image           whether to save original image
    :param save_prob_index:     The probability map of which class will be saved. Save no prob if setting to -1.
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
    for i, file_path in enumerate(file_path_list):
      print('{}: {}'.format(i, file_path))

      # load image
      begin = time.time()
      image = sitk.ReadImage(file_path, sitk.sitkFloat32)
      read_image_time = time.time() - begin

      # set iso mask and prob
      iso_image = resample_spacing(image, model['spacing'], model['interpolation'])
      iso_mask = sitk.Image(iso_image.GetSize(), sitk.sitkInt8)
      iso_mask.CopyInformation(iso_image)

      if save_prob_index >= 0:
        iso_mean_prob = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
        iso_mean_prob.CopyInformation(iso_image)

        iso_std_map = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
        iso_std_map.CopyInformation(iso_image)

      partition_type = model['infer_cfg'].general.partition_type
      if partition_type == 'DISABLE':
        start_voxels = [[0, 0, 0]]
        end_voxels = [[int(iso_image.GetSize()[idx]) for idx in range(3)]]

      elif partition_type == 'SIZE':
        partition_size = model['infer_cfg'].general.partition_size
        max_stride = model['max_stride']
        start_voxels, end_voxels = image_partition_by_fixed_size(iso_image, partition_size, max_stride)

      else:
        raise ValueError('Unsupported partition type!')

      begin = time.time()
      for idx in range(len(start_voxels)):
        start_voxel, end_voxel = start_voxels[idx], end_voxels[idx]

        voi_mask, voi_mean_prob, voi_std_map = \
          segmentation_voi(model, iso_image, start_voxel, end_voxel, gpu_id > 0, save_prob_index)
        iso_mask = copy_image(voi_mask, start_voxel, end_voxel, iso_mask)

        if save_prob_index >= 0:
          iso_mean_prob = copy_image(voi_mean_prob, start_voxel, end_voxel, iso_mean_prob)
          iso_std_map = copy_image(voi_std_map, start_voxel, end_voxel, iso_std_map)

        print('{:0.2f}%'.format((idx + 1) / len(start_voxels) * 100))
      test_time = time.time() - begin

      case_name = file_name_list[i]
      if not os.path.isdir(os.path.join(output_folder, case_name)):
        os.makedirs(os.path.join(output_folder, case_name))

      begin = time.time()
      # resample mask to the original spacing
      mask = resample(iso_mask, image, interp_method='NN')

      # pick the largest component
      # TO BE DONE
      post_processing_time = time.time() - begin

      begin = time.time()
      # save results
      sitk.WriteImage(mask, os.path.join(output_folder, case_name, seg_name))

      if save_image:
        sitk.WriteImage(image, os.path.join(output_folder, case_name, 'org.mha'), True)

      if save_prob_index >= 0:
        mean_prob = resample(iso_mean_prob, image, interp_method='LINEAR')
        mean_prob_save_path = os.path.join(output_folder, case_name, 'mean_prob_{}.mha'.format(save_prob_index))
        sitk.WriteImage(mean_prob, mean_prob_save_path, True)

        std_map = resample(iso_std_map, image, interp_method='LINEAR')
        std_map_save_path = os.path.join(output_folder, case_name, 'std_prob_{}.mha'.format(save_prob_index))
        sitk.WriteImage(std_map, std_map_save_path, True)
      save_time = time.time() - begin

      total_test_time = load_model_time + read_image_time + test_time + post_processing_time + save_time
      print('total test time: {:.2f}'.format(total_test_time))


def main():

    long_description = 'Inference engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Single image\n' \
                       '2. A text file containing paths of all testing images\n'\
                       '3. A folder containing all testing images\n'
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='/home/qinliu/debug/org.mha',
                        help='input folder/file for intensity images')
    parser.add_argument('-m', '--model',
                        default='/home/qinliu/debug/model_0129_2020',
                        help='model root folder')
    parser.add_argument('-o', '--output',
                        default='/home/qinliu/debug/results',
                        help='output folder for segmentation')
    parser.add_argument('-n', '--seg_name',
                        default='result.mha',
                        help='the name of the segmentation result to be saved')
    parser.add_argument('-g', '--gpu_id', type=int,
                        default=-1,
                        help='the gpu id to run model, set to -1 if using cpu only.')
    parser.add_argument('--save_image',
                        help='whether to save original image', action="store_true")
    parser.add_argument('--save_prob_index', type=int,
                        default=1,
                        help='whether to save single prob map')
    args = parser.parse_args()

    segmentation(args.input, args.model, args.output, args.seg_name, args.gpu_id, args.save_image,
                 args.save_prob_index)


if __name__ == '__main__':
    main()