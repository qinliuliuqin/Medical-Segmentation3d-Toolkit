import argparse
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk


def cal_dsc_batch():
  """ Batch test for calculating dice ratio """

  num_classes = 3
  test_file = '/shenlab/lab_stor6/qinliu/CT_Dental/datasets/test.txt'
  segmentation_folder = '/shenlab/lab_stor6/qinliu/CT_Dental/results/model_1112_2019_epoch3700'
  ground_truth_folder = '/shenlab/lab_stor6/qinliu/CT_Dental/data'
  result_file = '/shenlab/lab_stor6/qinliu/CT_Dental/results/results_dsc_epoch3700.csv'

  file_list, case_list = read_test_txt(test_file)

  result_content = []
  for case_name in case_list:
    print(case_name)

    gt_path = os.path.join(ground_truth_folder, case_name, 'seg.mha')
    gt = cio.read_image(gt_path)

    seg_path = os.path.join(segmentation_folder, case_name, 'seg.mha')
    seg = cio.read_image(seg_path)

    # calculate dsc for each label
    content = [case_name]
    for label in range(1, num_classes):
      gt_copy, seg_copy = gt.deep_copy(), seg.deep_copy()
      ctools.convert_multi_label_to_binary(gt_copy, label)
      ctools.convert_multi_label_to_binary(seg_copy, label)
      score, type = cal_dsc_binary(gt_copy.to_numpy(), seg_copy.to_numpy())
      content.extend([score, type])

    result_content.append(content)

  column = ['filename']
  for label in range(1, num_classes):
    column.extend(['label{}_score'.format(label), 'label{}_type'.format(label)])

  df = pd.DataFrame(data=result_content, columns=column)
  df.to_csv(result_file)




if __name__ == '__main__':
  long_description = "Evaluation code for 3d medical image segmentation"
  parser = argparse.ArgumentParser(description=long_description)

  parser.add_argument('-i', '--input',
                      default='./config/train_config.py',
                      help='configure file for medical image segmentation training.')
  args = parser.parse_args()
