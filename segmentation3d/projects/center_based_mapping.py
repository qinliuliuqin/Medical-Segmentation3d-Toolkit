import torch
import torch.nn as nn


class product_based_mapping_layer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(product_based_mapping_layer, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, in_tensor):
        out_tensor = self.conv1(in_tensor)
        return out_tensor


class center_based_mapping_layer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(center_based_mapping_layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center = nn.Parameter(nn.init.normal_(torch.zeros([1, 1, out_channels, in_channels])), requires_grad=True)
        self.dim_scale = nn.Parameter(torch.exp(nn.init.uniform_(torch.zeros([1, 1, 1, in_channels]), -0.1, 0.1)), requires_grad=True)
        self.cls_scale = nn.Parameter(nn.init.uniform_(torch.zeros([1, 1, out_channels]), -0.1, 0.1), requires_grad=True)

    def forward(self, in_tensor):
        assert isinstance(in_tensor, torch.Tensor) and in_tensor.dim() == 5
        assert in_tensor.shape[1] == self.in_channels

        in_shape = in_tensor.shape
        in_tensor = in_tensor.permute(dims=[0,2,3,4,1]).view(in_shape[0], -1, 1, in_shape[1])
        out_tensor = self.cls_scale * torch.sqrt(1e-6 + ((in_tensor - self.center) * (in_tensor - self.center) * self.dim_scale).mean(dim=-1))

        return out_tensor.view(in_shape[0], self.out_channels, in_shape[2], in_shape[3], in_shape[4])


if __name__ == '__main__':

    batch, in_ch, out_ch, dim_z, dim_y, dim_x = 1, 32, 16, 8, 8, 8
    in_tensor = torch.randn(batch, in_ch, dim_z, dim_y, dim_x)

    c_layer = center_based_mapping_layer(in_ch, out_ch)
    c_out = c_layer(in_tensor)

    p_layer = product_based_mapping_layer(in_ch, out_ch)
    p_out = p_layer(in_tensor)

    assert c_out.shape == p_out.shape
    assert c_out.shape[0] == batch and c_out.shape[1] == out_ch
    assert c_out.shape[2] == dim_z and c_out.shape[3] == dim_y and c_out.shape[4] == dim_x
    assert sum(p.numel() for p in c_layer.parameters()) == in_ch * out_ch + out_ch + in_ch
    assert sum(p.numel() for p in p_layer.parameters()) == in_ch * out_ch + out_ch
