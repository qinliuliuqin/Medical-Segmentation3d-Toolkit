import numpy as np
import SimpleITK as sitk


def cal_dsc(gt_npy, seg_npy, label, threshold):
  """ Calculate dice ratio
  :param gt_npy: the input ground truth.
  :param seg_npy: the input segmentation result.
  :param label: the label for dsc calculation.
  :param threshold: the segmentation threshold, only the number of voxels greater
                    than the threshold will be regarded as valid segmentation.
  return:
    dsc: the dice similarity coefficient or dice ratio.
    seg_type: the segmentation type.
  """
  if isinstance(gt_npy, sitk.Image):
    gt_npy = sitk.GetArrayFromImage(gt_npy)
  
  if isinstance(seg_npy, sitk.Image):
    seg_npy = sitk.GetArrayFromImage(seg_npy)
  
  gt_npy, seg_npy = (gt_npy == label), (seg_npy == label)
  area_gt, area_seg = np.sum(gt_npy), np.sum(seg_npy)
  
  if area_gt < threshold and area_seg < threshold:
    dsc, seg_type = 1.0, 'TN'
  
  elif area_gt < threshold and area_seg >= threshold:
    dsc, seg_type = 0.0, 'FP'
  
  elif area_gt >= threshold and area_seg < threshold:
    dsc, seg_type = 0.0, 'FN'
  
  else:
    intersection = np.sum(gt_npy & seg_npy)
    dsc, seg_type = 2 * intersection / (area_gt + area_seg), 'TP'
  
  return dsc, seg_type