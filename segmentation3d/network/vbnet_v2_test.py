import torch

from segmentation3d.network.vbnet_v2 import SegmentationNet
from segmentation3d.network.module.init import kaiming_weight_init


def test_vbnet():

  in_channel = 1
  out_channel = 16

  batch_size = 1
  dim_x, dim_y, dim_z = 32, 32, 32

  in_tensor = torch.randn([batch_size, in_channel, dim_z, dim_y, dim_x])

  net = SegmentationNet(in_channel, out_channel)
  net.apply(kaiming_weight_init)
  out_tensor = net(in_tensor)

  print(out_tensor.shape)


if __name__ == '__main__':

  test_vbnet()