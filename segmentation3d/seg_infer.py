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
from segmentation3d.dataloader.image_tools import get_image_frame, set_image_frame, crop_image, \
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


def segmentation_roi(model, iso_image, start_voxel, end_voxel, use_gpu, iter):
  assert isinstance(iso_image, sitk.Image)

  roi_image = iso_image[start_voxel[0]:end_voxel[0], start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2]]

  if model['crop_normalizers'] is not None:
    roi_image = model.crop_normalizers[0](roi_image)

  roi_image_tensor = convert_image_to_tensor(roi_image).unsqueeze(0)
  if use_gpu:
    roi_image_tensor = roi_image_tensor.cuda()

  with torch.no_grad():
    probs = model['net'](roi_image_tensor, 'test')
    probs = torch.unsqueeze(probs, 0)
    for i in range(iter - 1):
      probs = torch.cat((probs, torch.unsqueeze(model['net'](roi_image_tensor, 'test'), 0)), 0)
    mean_probs, stddev_probs = torch.mean(probs, 0), torch.std(probs, 0)

  # return segmentation mask
  _, mask = mean_probs.max(1)
  mask = convert_tensor_to_image(mask[0].data, dtype=np.int8)
  mask.CopyInformation(roi_image)

  # return the average probability map
  mean_prob = None
  save_prob_index = int(model['save_prob_index'])
  if save_prob_index >= 0:
    mean_prob = convert_tensor_to_image(mean_probs[0][save_prob_index].data, dtype=np.float)
    mean_prob.CopyInformation(roi_image)

  # return the standard deviation of probability map
  uncertainty_map = None
  if save_prob_index >= 0:
    uncertainty_map = convert_tensor_to_image(stddev_probs[0][save_prob_index].data, dtype=np.float)
    uncertainty_map.CopyInformation(roi_image)

  return mask, mean_prob, uncertainty_map


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

    begin = time.time()
    # load model
    model = load_seg_model(model_folder, gpu_id)
    model.save_image = save_image
    model.save_prob_index = save_prob_index
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

      # set mask and prob
      mask = sitk.Image(image.GetSize(), sitk.sitkInt8)
      mask.CopyInformation(image)

      if model.save_prob_index >= 0:
        prob = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        prob.CopyInformation(image)

        std_prob = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        std_prob.CopyInformation(image)

      # set iso mask and prob
      iso_image = resample_spacing(image, model['spacing'], model['interpolation'])

      iso_mask = sitk.Image(iso_image.GetSize(), sitk.sitkInt8)
      iso_mask.CopyInformation(iso_image)

      if model.save_prob_index >= 0:
        iso_prob = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
        iso_prob.CopyInformation(iso_image)

        iso_uncertainty = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
        iso_uncertainty.CopyInformation(iso_image)

      begin = time.time()
      partition_type = model['infer_cfg'].general.partition_type
      if partition_type == 'DISABLE':
        start_voxel = [0, 0, 0]
        end_voxel = [int(iso_image.GetSize()[idx]) for idx in range(3)]
        roi_mask, roi_prob, roi_uncertainty = \
          segmentation_roi(model, iso_image, start_voxel, end_voxel, gpu_id > 0, 5)

        iso_mask = copy_image(roi_mask, start_voxel, end_voxel, iso_mask)

      if model.save_prob_index >= 0:
        iso_uncertainty = copy_image(roi_uncertainty, start_voxel, end_voxel, iso_uncertainty)

      elif partition_type == 'SIZE':
        partition_size = model['infer_cfg'].general.partition_size
        max_stride = model['max_stride']
        start_voxels, end_voxels = image_partition_by_fixed_size(iso_image, partition_size, max_stride)

        for idx in range(len(start_voxels)):
          start_voxel, end_voxel = start_voxels[idx], end_voxels[idx]

          roi_mask, roi_prob, roi_uncertainty = \
            segmentation_roi(model, iso_image, start_voxel, end_voxel, gpu_id > 0, 5)

          iso_mask = copy_image(roi_mask, start_voxel, end_voxel, iso_mask)

          if model.save_prob_index >= 0:
            iso_uncertainty = copy_image(roi_uncertainty, start_voxel, end_voxel, iso_uncertainty)

          print('{:0.2f}%'.format((idx + 1) / len(start_voxels) * 100))

      else:
        raise ValueError('Unsupported partition type!')
      test_time = time.time() - begin

      case_name = file_name_list[i]
      if not os.path.isdir(os.path.join(output_folder, case_name)):
        os.makedirs(os.path.join(output_folder, case_name))

      begin = time.time()
      # save results
      if model.save_image:
        sitk.WriteImage(image, os.path.join(output_folder, case_name, 'org.mha'), True)

      # if model.save_prob_index >= 0:
      #   sitk.WriteImage(prob, os.path.join(output_folder, case_name, 'prob_{}.mha'.format(model.save_prob_index)), True)
      #   sitk.WriteImage(std_prob, os.path.join(output_folder, case_name,
      #                                          'uncertainty_{}.mha'.format(model.save_prob_index)), True)

      sitk.WriteImage(iso_mask, os.path.join(output_folder, case_name, seg_name), True)
      save_time = time.time() - begin

      if model.save_prob_index >= 0:
        sitk.WriteImage(iso_uncertainty, os.path.join(output_folder, case_name,
                                                      'uncertainty_{}.mha'.format(model.save_prob_index)), True)

      total_test_time = load_model_time + read_image_time + test_time + save_time
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
                        default='/home/qinliu/debug/model_0128_2020_focal',
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