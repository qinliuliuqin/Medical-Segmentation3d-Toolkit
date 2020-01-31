import numpy as np
import SimpleITK as sitk


def cal_dsc(ground_truth, segmentation, label, threshold):
  """ Calculate dice ratio

  :param ground_truth: the input ground truth.
  :param segmentation: the input segmentation result.
  :param label: the label for dsc calculation.
  :param threshold: the segmentation threshold.
  return:
    dsc: the dice similarity coefficient or dice ratio.
    seg_type: the segmentation type.
  """
  assert isinstance(ground_truth, sitk.Image)
  assert isinstance(segmentation, sitk.Image)

  gt_npy = sitk.GetArrayFromImage(ground_truth)
  seg_npy = sitk.GetArrayFromImage(segmentation)

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