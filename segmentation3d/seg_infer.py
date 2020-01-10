import argparse
import edict
import importlib
import os
import SimpleITK as sitk
import time
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict as edict

from segmentation3d.utils.file_io import load_config
from segmentation3d.utils.model_io import get_checkpoint_folder
from segmentation3d.dataloader.image_tools import get_image_frame, set_image_frame, crop_image, \
  convert_image_to_tensor, convert_tensor_to_image


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
  state = torch.load(chk_file)

  # load network module
  net_module = importlib.import_module('segmentation3d.network.' + state['net'])
  net = net_module.SegmentationNet(state['in_channels'], state['out_channels'])
  net.load_state_dict(state['state_dict'])
  net.eval()

  if gpu_id >= 0:
    net = nn.parallel.DataParallel(net)
    net = net.cuda()

  model.net = net
  model.spacing = state['spacing']
  model.max_stride = state['max_stride']
  model.interpolation = state['interpolation']

  return model


def segmentation_voi(image, model, center, size):
  """ Segment a volume of interest from an image. The volume will be cropped from the image first with the specified
  center and size, and then, the cropped block will be segmented by the segmentation model.

  :param image: The input image
  :param model: The segmentation model
  :param center: The volume center in world coordinate, unit: mm
  :param size:   The volume size in physical space, unit: mm
  """
  assert isinstance(image, sitk.Image)

  # the cropping size should be multiple of the max_stride
  max_stride = model['max_stride']
  cropping_size = [int(size[idx] / model['spacing'][idx]) for idx in range(3)]
  for idx in range(3):
    if cropping_size[idx] % max_stride:
      cropping_size[idx] += max_stride - cropping_size[idx] % max_stride

  iso_image = crop_image(image, center, cropping_size, model['spacing'], model['interpolation'])
  iso_image_tensor = convert_image_to_tensor(iso_image).unsqueeze(0)

  with torch.no_grad():
    probs = model['net'](iso_image_tensor)

  # return segmentation mask
  _, mask = probs.max(1)
  mask = convert_tensor_to_image(mask[0].data, dtype=np.short)
  set_image_frame(mask, get_image_frame(iso_image))

  # return probability map
  prob_map = None
  save_prob_index = int(model['save_prob_index'])
  if save_prob_index >= 0:
    prob_map = convert_tensor_to_image(probs[0][save_prob_index].data, dtype=np.float)
    set_image_frame(prob_map, get_image_frame(iso_image))

  return mask, prob_map


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
    total_test_time = 0

    # load model
    model = load_seg_model(model_folder, gpu_id)
    model.save_image = save_image
    model.save_prob_index = save_prob_index

    # load image
    image = sitk.ReadImage(input_path)

    partition_type = model['infer_cfg'].general.partition_type
    if partition_type == 'DISABLE':
      center_voxel = [float(image.GetSize()[idx] / 2.0) for idx in range(3)]
      center_world = image.TransformContinuousIndexToPhysicalPoint(center_voxel)
      volume_size = [32, 32, 32]
      mask, prob = segmentation_voi(image, model, center_world, volume_size)

      # debug only
      mask_path = '/home/qinliu/mask.mha'
      prob_path = '/home/qinliu/prob.mha'
      sitk.WriteImage(mask, mask_path, True)
      sitk.WriteImage(prob, prob_path, True)

    elif partition_type == 'NUM':
      pass

    elif partition_type == 'SIZE':
      pass

    else:
      raise ValueError('Unsupported partition type!')


    # save segmentation mask
    pass


def main():

    long_description = 'Inference engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Single image\n' \
                       '2. A text file containing paths of all testing images\n' \
                       '3. A folder containing all testing images\n'
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='/home/qinliu/projects/segmentation3d/debug/org.mha',
                        help='input folder/file for intensity images')
    parser.add_argument('-m', '--model',
                        default='/home/qinliu/projects/segmentation3d/debug/model_0110_2020',
                        help='model root folder')
    parser.add_argument('-o', '--output',
                        default='',
                        help='output folder for segmentation')
    parser.add_argument('-n', '--seg_name',
                        default='result.mha',
                        help='the name of the segmentation result to be saved')
    parser.add_argument('-g', '--gpu_id',
                        default='-1',
                        help='the gpu id to run model, set to -1 if using cpu only.')
    parser.add_argument('--save_image',
                        help='whether to save original image', action="store_true")
    parser.add_argument('--save_prob_index',
                        default='1',
                        help='whether to save single prob map')
    args = parser.parse_args()

    segmentation(args.input, args.model, args.output, args.seg_name, int(args.gpu_id), args.save_image,
                 args.save_prob_index)


if __name__ == '__main__':
    main()
