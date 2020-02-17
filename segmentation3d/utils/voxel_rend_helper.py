from torch.nn import functional as F

"""
Shape shorthand in this module:
    N: mini-batch dimension size, i.e. the number of RoIs for instance segmentation or the
        number of images for semantic segmentation.
    R: number of ROIs, combined over all images, in the mini-batch
    P: number of voxels
"""


def voxel_sample(input, voxel_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `voxel_coords` to lie inside
    [0, 1] x [0, 1] x [0, 1] cubic.
    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        voxel_coords (Tensor): A tensor of shape (N, P, 2) or (N, D_grid, H_grid, W_grid, 2) that contains
        [0, 1] x [0, 1] x [0, 1] normalized voxel coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if voxel_coords.dim() == 3:
      add_dim = True
      voxel_coords = voxel_coords.unsqueeze(2).unsqueeze(2)

    output = F.grid_sample(input, 2.0 * voxel_coords - 1.0, **kwargs)
    if add_dim:
      output = output.squeeze(2).squeeze(2)

    return output