from segmentation3d.network.vbnet_rend import *


def test_voxel_head():
  num_batches, num_fine_channels, num_coarse_channels, num_voxels = 2, 80, 3, 128
  features_fine = torch.randn(num_batches, num_fine_channels, num_voxels)
  features_coarse = torch.randn(num_batches, num_coarse_channels, num_voxels)

  num_out_channels, num_fc = 9, 4
  net = VoxelHead(num_fine_channels, num_coarse_channels, num_out_channels, num_fc)
  pred = net(features_fine, features_coarse)

  assert pred.dim() == 3
  assert pred.shape[0] == num_batches
  assert pred.shape[1] == num_out_channels
  assert pred.shape[2] == num_voxels


if __name__ == '__main__':
  test_voxel_head()