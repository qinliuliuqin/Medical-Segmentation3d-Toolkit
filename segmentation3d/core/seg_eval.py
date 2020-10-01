import os
import pandas as pd
import SimpleITK as sitk

from segmentation3d.core.seg_infer import read_test_csv
from segmentation3d.utils.metrics import cal_dsc


def evaluation(eval_file, seg_folder, seg_name, labels, threshold, save_csv_file_path):
    """ Evaluation """
    assert eval_file.endswith('.csv')

    image_name_list, _, mask_path_list = read_test_csv(eval_file, mode='eval')
    seg_path_list = []
    for case_name in image_name_list:
        seg_path_list.append(os.path.join(seg_folder, case_name, seg_name))

    return cal_dsc_batch(mask_path_list, seg_path_list, image_name_list, labels, threshold, save_csv_file_path)


def cal_dsc_batch(gt_files, seg_files, case_names, labels, threshold, save_csv_file_path):
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
        case_name = case_names[idx]
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
    mean_a, std_a = 0, 0
    for label in labels:
        mean, std = df['label{}_score'.format(label)].mean(), \
                    df['label{}_score'.format(label)].std()
        print(mean, std)
        mean_a += mean
        std_a += std
        statistics_content[0].extend([mean, 'ignore_type'])
        statistics_content[1].extend([std, 'ignore_type'])

    if save_csv_file_path is not None:
        df_statistics = pd.DataFrame(data=statistics_content, columns=column)
        df = df.append(df_statistics)
        df.to_csv(save_csv_file_path)

    return mean_a / len(labels), std_a / len(labels)