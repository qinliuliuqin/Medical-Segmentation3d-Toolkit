import torch
from torch.nn import functional as F

"""
Shape shorthand in this module:
    N: mini-batch dimension size, i.e. the number of RoIs for instance segmentation or the
        number of images for semantic segmentation.
    R: number of ROIs, combined over all images, in the mini-batch
    P: number of voxels
"""


def voxel_sample(input, voxel_coords, mode='bilinear', padding_mode='zeros'):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `voxel_coords` to lie inside
    [0, 1] x [0, 1] x [0, 1] cubic.
    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        voxel_coords (Tensor): A tensor of shape (N, P, 3) or (N, D_out, H_out, W_out, 3) that contains
        [0, 1] x [0, 1] x [0, 1] normalized voxel coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, D_out * H_out * W_out) that contains features for
            points in `voxel_coords`. The features are obtained via bilinear interpolation from `input` the same way
            as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if voxel_coords.dim() == 3:
      add_dim = True
      voxel_coords = voxel_coords.unsqueeze(2).unsqueeze(2)

    output = F.grid_sample(input, 2.0 * voxel_coords - 1.0, mode=mode, padding_mode=padding_mode)
    if add_dim:
      output = output.squeeze(3).squeeze(3)

    if output.dim() == 5:
      output = output.view(output.shape[0], output.shape[1], -1)

    return output


def calculate_uncertainty(preds):
  """
  We estimate uncertainty as L1 distance between 0.0 and the prediction in probability for the
      foreground class in `classes`.
  Args:
      preds (Tensor): A tensor of shape (R, C, ...) or (R, 2, ...), where R is the total number of selected voxels
          and C is the number of classes.
      classes (list): A list of length R that contains either predicted of ground truth class
          for eash predicted mask.
  Returns:
      scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
          the most uncertain locations having the highest uncertainty score.
  """
  gt_class_preds, _ = preds.max(dim=1)
  return - gt_class_preds.unsqueeze(dim=1)


def get_uncertain_voxel_coords_with_randomness(
    preds, uncertainty_func, num_voxels, oversample_ratio, importance_sample_ratio):
  """
  Sample points in [0, 1] x [0, 1] x [0, 1] coordinate space based on their uncertainty. The uncertainties
      are calculated for each point using 'uncertainty_func' function that takes point's prediction as input.
  Args:
      preds (Tensor): A tensor of shape (N, C, D, H, W).
      uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 2, P) that contains predictions for P
          points and returns their uncertainties as a Tensor of shape (N, 1, P).
      num_voxels (int): The number of voxels P to sample.
      oversample_ratio (int): Oversampling parameter.
      importance_sample_ratio (float): Ratio of points that are sampled via importance sampling.
  Returns:
      voxel_coords (Tensor): A tensor of shape (N, P, 3) that contains the coordinates of P sampled points.
  """
  assert oversample_ratio >= 1
  assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0

  num_batches = preds.shape[0]
  num_sampled = int(num_voxels * oversample_ratio)
  voxel_coords = torch.rand(num_batches, num_sampled, 3, device=preds.device)
  voxel_preds = voxel_sample(preds, voxel_coords)
  voxel_uncertainties = uncertainty_func(voxel_preds)
  num_uncertain_voxels = int(importance_sample_ratio * num_voxels)
  num_random_voxels = num_voxels - num_uncertain_voxels
  idx = torch.topk(voxel_uncertainties[:, 0, :], k=num_uncertain_voxels, dim=1)[1]
  shift = num_sampled * torch.arange(num_batches, dtype=torch.long, device=preds.device)
  idx += shift[:, None]
  voxel_coords = voxel_coords.view(-1, 3)[idx.view(-1), :].view(num_batches, num_uncertain_voxels, 3)
  if num_random_voxels > 0:
    voxel_coords = torch.cat([
        voxel_coords,
        torch.rand(num_batches, num_random_voxels, 3, device=preds.device),
      ],
      dim=1,
    )
  return voxel_coords


def voxel_sample_features(features_list, voxel_coords):
  """
  Get features from feature maps in `features_list` that correspond to the specified voxel coordinates.
    Args:
      features_list (list[Tensor]): A list of feature map tensors to get features from. Each feature map in the list
          is a tensor of shape (R, :, D, H, W). Different feature maps have different channels.
      voxel_coords (Tensor): A tensor of shape (R, P, 3) that contains [0, 1] x [0, 1] x [0, 1] normalized coordinates
          of the P sampled points.
  Returns:
      point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled from all features maps
          in feature_list for P sampled points for all R boxes in `boxes`.
  """

  voxel_features = []
  for idx_feature, feature_map in enumerate(features_list):
    voxel_features.append(voxel_sample(feature_map, voxel_coords))

  return torch.cat(voxel_features, dim=1)