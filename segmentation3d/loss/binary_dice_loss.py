import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """ Dice Loss for binary segmentation
    """

    def forward(self, input, target):
        batchsize = input.size(0)

        # convert probability to binary label using maximum probability
        input_pred, input_label = input.max(1)
        input_pred *= input_label

        # convert to floats
        input_pred = input_pred.float()
        target_label = target.float()

        # convert to 1D
        input_pred = input_pred.view(batchsize, -1)
        target_label = target_label.view(batchsize, -1)

        # compute dice score
        intersect = torch.sum(input_pred * target_label, 1)
        input_area = torch.sum(input_pred * input_pred, 1)
        target_area = torch.sum(target_label * target_label, 1)

        sum = input_area + target_area
        epsilon = torch.tensor(1e-6)

        # batch dice loss and ignore dice loss where target area = 0
        batch_loss = torch.tensor(1.0) - (torch.tensor(2.0) * intersect + epsilon) / (sum + epsilon)
        loss = batch_loss.mean()

        return loss