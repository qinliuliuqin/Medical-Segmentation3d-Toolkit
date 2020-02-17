from segmentation3d.utils.voxel_rend_helper import *


def test_voxel_sample():
  num_batches, num_classes, dim_z, dim_y, dim_x = 2, 3, 4, 4, 4
  input = torch.rand(num_batches, num_classes, dim_z, dim_y, dim_x)

  # test case 1
  num_voxels = 16
  voxel_coords = torch.rand(num_batches, num_voxels, 3)
  output = voxel_sample(input, voxel_coords)
  assert output.shape[0] == num_batches and output.shape[1] == num_classes and output.shape[2] == num_voxels

  # test case 2
  dim_z_out, dim_y_out, dim_x_out = 2, 2, 2
  voxel_coords = torch.rand(num_batches, dim_z_out, dim_y_out, dim_x_out, 3)
  output = voxel_sample(input, voxel_coords)
  assert output.shape[0] == num_batches and output.shape[1] == num_classes and \
         output.shape[2] == dim_z_out * dim_y_out * dim_x_out


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

  assert voxel_coords.shape[0] == num_batches and voxel_coords.shape[1] == num_voxels and voxel_coords.shape[2] == 3


def test_voxel_sample_features():
  num_batches, num_channels, dim_z, dim_y, dim_x = 2, 3, 8, 8, 8
  feature_map_fine = torch.rand(num_batches, num_channels, dim_z, dim_y, dim_x)
  feature_map_coarse = torch.randn(num_batches, num_channels * 2, dim_z // 2, dim_y // 2, dim_x // 2)

  num_voxels = 16
  voxel_coords = torch.rand(num_batches, num_voxels, 3)
  feature_map_list = [feature_map_fine, feature_map_coarse]
  sampled_features = voxel_sample_features(feature_map_list, voxel_coords)

  assert sampled_features.shape[0] == num_batches and \
         sampled_features.shape[1] == sum([feature_map_list[idx].shape[1] for idx in range(len(feature_map_list))])

if __name__ == '__main__':

  test_get_uncertain_voxel_coords_with_randomness()

  test_voxel_sample()

  test_voxel_sample_features()