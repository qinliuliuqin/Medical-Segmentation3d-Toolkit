import os
import pandas as pd
import SimpleITK as sitk

from segmentation3d.utils.metrics import cal_dsc
from segmentation3d.seg_infer import read_test_txt


def cal_dsc_batch(gt_files, seg_files, labels, threshold, save_csv_file_path):
  """ Batch test for calculating dice ratio
  gt_files: a list containing all ground truth files.
  seg_files: a list containing all segmentation files.
  labels: a list containing which labels to calculate dice.
  threshold: an int value indicating the minimal number of the voxels for each
    label. If the number of voxels is lower than this value, then this label
    will be ignored.
  save_csv_file_path: the result file in csv format.
  """
  assert isinstance(gt_files, list) and isinstance(seg_files, list)
  assert len(gt_files) == len(seg_files)
  
  result_content = []
  for idx, gt_case_path in enumerate(gt_files):
    gt = sitk.ReadImage(gt_case_path)
    gt_npy = sitk.GetArrayFromImage(gt)
    
    seg_case_path = seg_files[idx]
    seg = sitk.ReadImage(seg_case_path)
    seg_npy = sitk.GetArrayFromImage(seg)
    
    # calculate dsc for each label
    case_name = os.path.basename(os.path.dirname(gt_case_path))
    content = [case_name]
    for label in labels:
      score, type = cal_dsc(gt_npy, seg_npy, label, threshold)
      content.extend([score, type])
      print('case_name: {}, label: {}, score: {}, type: {}'.format(
        case_name, label, score, type))
    
    result_content.append(content)
  
  column = ['filename']
  for label in labels:
    column.extend(['label{}_score'.format(label), 'label{}_type'.format(label)])
  df = pd.DataFrame(data=result_content, columns=column)
  
  statistics_content = [['mean'], ['std']]
  for label in labels:
    mean, std = df['label{}_score'.format(label)].mean(), \
                df['label{}_score'.format(label)].std()
    print(mean, std)
    statistics_content[0].extend([mean, 'ignore_type'])
    statistics_content[1].extend([std, 'ignore_type'])
  
  df_statistics = pd.DataFrame(data=statistics_content, columns=column)
  df = df.append(df_statistics)
  df.to_csv(save_csv_file_path)


def test_cal_dsc_batch():
  test_file = '/shenlab/lab_stor6/qinliu/CT_Dental/datasets/test.txt'
  gt_folder = '/shenlab/lab_stor6/qinliu/CT_Dental/data'
  gt_name = 'seg.mha'
  seg_folder = '/shenlab/lab_stor6/qinliu/CT_Dental/results/model_0227_2020/model1_master_0.4_contrast'
  seg_name = 'seg.mha'
  result_file = '/shenlab/lab_stor6/qinliu/CT_Dental/results/model_0227_2020/model1_master_0.4_contrast/results.csv'
  
  file_list, case_list = read_test_txt(test_file)
  gt_files = []
  for case_name in file_list:
    gt_files.append(os.path.join(gt_folder, case_name, gt_name))
  
  seg_files = []
  for case_name in file_list:
    seg_files.append(os.path.join(seg_folder, case_name, seg_name))
  
  labels = [1, 2]
  cal_dsc_batch(gt_files, seg_files, labels, 10, result_file)


if __name__ == '__main__':
  test_cal_dsc_batch()
