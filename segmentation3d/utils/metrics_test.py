import SimpleITK as sitk
from segmentation3d.utils.metrics import cal_dsc


seg_path = '/home/qinliu/debug/seg.mha'
gt_path = '/home/qinliu/debug/seg.mha'

seg = sitk.ReadImage(seg_path)
gt = sitk.ReadImage(gt_path)

dsc, seg_type = cal_dsc(gt, seg, 0, 10)
print(dsc, seg_type)