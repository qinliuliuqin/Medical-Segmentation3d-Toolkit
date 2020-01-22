import numpy as np
import SimpleITK as sitk

from segmentation3d.utils.dicom_helper import read_dicom_series, write_dicom_series, dicom_tags_dict


if __name__ == '__main__':

  # read mha image
  seg_path = '/home/qinliu/debug/org.mha'
  seg = sitk.ReadImage(seg_path, sitk.sitkInt16)

  # save mha to dicom series
  tags = dicom_tags_dict()
  dicom_save_folder = '/home/qinliu/debug/org_dicom'
  write_dicom_series(seg, dicom_save_folder, tags=tags)

  # load the saved dicom series
  seg_reload = read_dicom_series(dicom_save_folder)
  seg_reload_path = '/home/qinliu/debug/org_reload.mha'
  sitk.WriteImage(seg_reload, seg_reload_path)

  # compare the original image and the reloaded image
  image_npy = sitk.GetArrayFromImage(seg)
  image_reloaded_npy = sitk.GetArrayFromImage(seg_reload)
  assert np.sum(np.abs(image_npy - image_reloaded_npy)) < 1e-6
