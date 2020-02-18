import torch
import torch.nn as nn

from segmentation3d.network.module.weight_init import kaiming_weight_init, gaussian_weight_init
from segmentation3d.network.module.vnet_inblock import InputBlock
from segmentation3d.network.module.vnet_outblock import OutputBlock
from segmentation3d.network.module.vnet_upblock import UpBlock
from segmentation3d.network.module.vnet_downblock import DownBlock


def parameters_kaiming_init(net):
    """ model parameters initialization """
    net.apply(kaiming_weight_init)


def parameters_gaussian_init(net):
    """ model parameters initialization """
    net.apply(gaussian_weight_init)


class SegmentationNet(nn.Module):
    """ volumetric segmentation network """

    def __init__(self, in_channels, out_channels, dropout_turn_on=False):
        super(SegmentationNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, compression=True)
        self.down_64 = DownBlock(32, 2, compression=True)
        self.down_128 = DownBlock(64, 3, compression=True)
        self.down_256 = DownBlock(128, 3, compression=True)
        self.up_256 = UpBlock(256, 256, 3, compression=True)
        self.up_128 = UpBlock(256, 128, 3, compression=True)
        self.up_64 = UpBlock(128, 64, 2, compression=True)
        self.out_block = OutputBlock(64, out_channels)
        self.dropout_turn_on = dropout_turn_on
        if self.dropout_turn_on:
            self.dropout = nn.Dropout3d(p=0.5, inplace=False)


    def forward(self, input):
        assert isinstance(input, torch.Tensor)

        if self.dropout_turn_on:
            self.dropout.train()

        l1_out16 = self.in_block(input)
        l2_out32 = self.down_32(l1_out16)

        l3_out64 = self.down_64(l2_out32)
        if self.dropout_turn_on:
            l3_out64 = self.dropout(l3_out64)

        l4_out128 = self.down_128(l3_out64)
        if self.dropout_turn_on:
            l4_out128 = self.dropout(l4_out128)

        l5_out256 = self.down_256(l4_out128)
        if self.dropout_turn_on:
            l5_out256 = self.dropout(l5_out256)

        r4_out256 = self.up_256(l5_out256, l4_out128)
        if self.dropout_turn_on:
            r4_out256 = self.dropout(r4_out256)

        r3_out128 = self.up_128(r4_out256, l3_out64)
        if self.dropout_turn_on:
            r3_out128 = self.dropout(r3_out128)

        r2_out64 = self.up_64(r3_out128, l2_out32)
        if self.dropout_turn_on:
            r2_out64 = self.dropout(r2_out64)

        out = self.out_block(r2_out64)

        return out, (l1_out16, r2_out64, r3_out128, r4_out256)

    def max_stride(self):
        return 16


class VoxelHead(nn.Module):
    """ Voxel classification network: A voxel head multi-layer perceptron which we model with conv1d layers with
        kernel 1. The head takes both multi-layer fine-grained and coarse prediction features as its input.
    """
    def __init__(self, in_fine_channels, in_coarse_channels, out_channels, num_fc):
        super(VoxelHead, self).__init__()

        fc_in_channels = in_fine_channels + in_coarse_channels
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_in_channels, in_fine_channels, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc_layers.append(fc)

        self.predictor = nn.Conv1d(fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, fine_features, coarse_features):
        x = torch.cat([fine_features, coarse_features], dim=1)
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
            x = torch.cat((x, coarse_features), dim=1)
        return self.predictor(x)
