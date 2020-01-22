import torch
import torch.nn as nn
from segmentation3d.network.module.residual_block3 import ResidualBlock3
from segmentation3d.network.module.weight_init import kaiming_weight_init, gaussian_weight_init


def parameters_kaiming_init(net):
    """ model parameters initialization """
    net.apply(kaiming_weight_init)


def parameters_gaussian_init(net):
    """ model parameters initialization """
    net.apply(gaussian_weight_init)


class DownBlock(nn.Module):
    """ downsample block of v-net """

    def __init__(self, in_channels, num_convs):
        super(DownBlock, self).__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.down_act = nn.ReLU(inplace=True)
        self.rblock = ResidualBlock3(out_channels, 3, 1, 1, num_convs)

    def forward(self, input):
        out = self.down_act(self.down_bn(self.down_conv(input)))
        out = self.rblock(out)
        return out


class UpBlock(nn.Module):
    """ Upsample block of v-net """

    def __init__(self, in_channels, out_channels, num_convs):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1)
        self.up_bn = nn.BatchNorm3d(out_channels // 2)
        self.up_act = nn.ReLU(inplace=True)
        self.rblock = ResidualBlock3(out_channels, 3, 1, 1, num_convs)

    def forward(self, input, skip):
        out = self.up_act(self.up_bn(self.up_conv(input)))
        out = torch.cat((out, skip), 1)
        out = self.rblock(out)
        return out


class InputBlock(nn.Module):
    """ input block of vb-net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class OutputBlock(nn.Module):
    """ output block of v-net

        The output is a list of foreground-background probability vectors.
        The length of the list equals to the number of voxels in the volume
    """

    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.act1(self.bn1(self.conv1(input)))
        out = self.bn2(self.conv2(out))
        out = self.softmax(out)
        return out


class SegmentationNet(nn.Module):
    """ volumetric segmentation network """

    def __init__(self, in_channels, out_channels, dropout_turn_on=False):
        super(SegmentationNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1)
        self.down_64 = DownBlock(32, 2)
        self.down_128 = DownBlock(64, 3)
        self.down_256 = DownBlock(128, 3)
        self.up_256 = UpBlock(256, 256, 3)
        self.up_128 = UpBlock(256, 128, 3)
        self.up_64 = UpBlock(128, 64, 2)
        self.up_32 = UpBlock(64, 32, 1)
        self.out_block = OutputBlock(32, out_channels)
        self.dropout_turn_on = dropout_turn_on
        if self.dropout_turn_on:
            self.dropout = nn.Dropout3d(p=0.5, inplace=False)


    def forward(self, input, mode='train'):

        if self.dropout_turn_on and mode == 'test':
            self.dropout.train()

        out16 = self.in_block(input)
        out32 = self.down_32(out16)

        out64 = self.down_64(out32)
        if self.dropout_turn_on:
            out64 = self.dropout(out64)

        out128 = self.down_128(out64)
        if self.dropout_turn_on:
            out128 = self.dropout(out128)

        out256 = self.down_256(out128)
        if self.dropout_turn_on:
            out256 = self.dropout(out256)

        out = self.up_256(out256, out128)
        if self.dropout_turn_on:
            out = self.dropout(out)

        out = self.up_128(out, out64)
        if self.dropout_turn_on:
            out = self.dropout(out)

        out = self.up_64(out, out32)
        if self.dropout_turn_on:
            out = self.dropout(out)

        out = self.up_32(out, out16)
        out = self.out_block(out)
        return out

    def max_stride(self):
        return 16
