from segmentation3d.utils.voxel_rend_helper import *


def test_get_uncertain_voxel_coords_with_randomness():
  num_batches, num_classes, dim_z, dim_y, dim_x = 2, 3, 4, 4, 4
  pred_coarse = torch.rand(num_batches, num_classes, dim_z, dim_y, dim_x)
  uncertain_func = calculate_uncertainty
  num_voxels = 16
  oversample_ratio = 2
  importance_sample_ratio = 0.75

  voxel_coords = get_uncertain_voxel_coords_with_randomness(
    pred_coarse, uncertain_func, num_voxels, oversample_ratio, importance_sample_ratio
  )

  print(voxel_coords.shape)


if __name__ == '__main__':

  test_get_uncertain_voxel_coords_with_randomness()