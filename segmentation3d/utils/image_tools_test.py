import SimpleITK as sitk
from segmentation3d.utils.image_tools import crop_image, resample_spacing, \
  pick_largest_connected_component, get_bounding_box


def test_copy_image():
  seg_path = '/home/qinliu/debug/seg.mha'
  seg = sitk.ReadImage(seg_path)

  assert isinstance(seg, sitk.Image)

  seg_empty = sitk.Image(seg.GetSize(), seg.GetPixelID())
  seg_empty.CopyInformation(seg)

  # crop from seg
  cropping_center_voxel = [int(seg.GetSize()[idx] // 2) for idx in range(3)]
  cropping_center_world = seg.TransformIndexToPhysicalPoint(cropping_center_voxel)
  cropping_size = [128, 128, 128]
  cropping_spacing = [1.0, 1.0, 1.0]
  interp_method = 'NN'
  seg_cropped = crop_image(seg, cropping_center_world, cropping_size, cropping_spacing, interp_method)

  seg_cropped_path = '/home/qinliu/debug/seg_cropped.mha'
  sitk.WriteImage(seg_cropped, seg_cropped_path)

  # copy_image(seg_cropped, cropping_center_world, cropping_size, seg_empty)
  # seg_copy_path = '/home/qinliu/debug/seg_empty_copy.mha'
  # sitk.WriteImage(seg_empty, seg_copy_path)

  seg_origin = seg.GetOrigin()
  seg_empty_origin = list(map(int, seg_empty.GetOrigin()))
  seg_cropped_size = list(map(int, seg_cropped.GetSize()))
  seg_cropped_origin = list(map(int, seg_cropped.GetSize()))
  seg_pasted = sitk.Paste(seg_empty, seg_cropped, seg_cropped_size, [0, 0, 0], [100, 100, 100])
  seg_paste_path = '/home/qinliu/debug/seg_empty_paste.mha'
  sitk.WriteImage(seg_pasted, seg_paste_path)


def test_resample_spacing():
  seg_path = '/home/qinliu/debug/org.mha'
  seg = sitk.ReadImage(seg_path)

  resampled_seg = resample_spacing(seg, [0.5, 0.5, 0.5], 'LINEAR')
  resampled_seg_path = '/home/qinliu/debug/resampled_seg.mha'
  sitk.WriteImage(resampled_seg, resampled_seg_path)


def crop_patch():
  image_path = '/home/qinliu/debug/org.mha'
  seg_path = '/home/qinliu/debug/seg.mha'

  image = sitk.ReadImage(image_path)
  seg = sitk.ReadImage(seg_path)

  cropping_center = [32, 32, 32]
  cropping_size = [32, 32, 32]
  cropping_spacing = [2.0, 2.0, 2.0]
  cropped_image = crop_image(image, cropping_center, cropping_size, cropping_spacing, 'LINEAR')
  cropped_seg = crop_image(seg, cropping_center, cropping_size, cropping_spacing, 'NN')

  cropped_image_path = '/home/qinliu/debug/cropped_org.mha'
  cropped_seg_path = '/home/qinliu/debug/cropped_seg.mha'
  sitk.WriteImage(cropped_image, cropped_image_path, True)
  sitk.WriteImage(cropped_seg, cropped_seg_path, True)


def test_pick_largest_connected_component():

    seg =sitk.ReadImage('/home/qinliu/debug/ROI_mask.nii.gz')
    threshold = 100000
    labels = [1, 2]

    seg_cc_path = '/home/qinliu/debug/seg_cc.mha'
    seg_cc = pick_largest_connected_component(seg, labels, threshold)
    sitk.WriteImage(seg_cc, seg_cc_path, True)


def test_get_bounding_box():

  seg_path = '/mnt/projects/CT_Pancreas/label/label0001.nii.gz'
  seg = sitk.ReadImage(seg_path)

  bbox_start, bbox_end = get_bounding_box(seg, None)
  print(bbox_start, bbox_end)

  seg_bbox_mask_npy = sitk.GetArrayFromImage(seg)
  seg_bbox_mask_npy[bbox_start[2]:bbox_end[2], bbox_start[1]:bbox_end[1], bbox_start[0]:bbox_end[0]] = 1
  seg_bbox_mask = sitk.GetImageFromArray(seg_bbox_mask_npy)
  seg_bbox_mask.CopyInformation(seg)

  seg_bbox_mask_path = '/home/ql/debug/bbox_mask.nii.gz'
  sitk.WriteImage(seg_bbox_mask, seg_bbox_mask_path)


if __name__ == '__main__':

  # test_copy_image()

  # test_resample_spacing()

  #crop_patch()

  #test_pick_largest_connected_component()

  test_get_bounding_box()