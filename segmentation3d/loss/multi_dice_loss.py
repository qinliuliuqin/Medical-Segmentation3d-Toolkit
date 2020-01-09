import torch
import torch.nn as nn
from segmentation3d.loss.binary_dice_loss import BinaryDiceLoss


class MultiDiceLoss(nn.Module):
    """ Dice Loss for multi-class segmentation
    """
    def __init__(self, weights, num_class, use_gpu):
        """
        :param weights: weight for each class dice loss
        :param num_class: the number of class
        """
        super(MultiDiceLoss, self).__init__()
        self.num_class = num_class

        assert len(weights) == self.num_class, "the length of weight must equal to num_class"
        self.weights = torch.FloatTensor(weights)
        self.weights = self.weights / self.weights.sum()

        if use_gpu:
            self.weights = self.weights.cuda()

    def forward(self, input_tensor, target):
        """
        :param input_tensor: network output tensor
        :param target: ground truth
        :return: weighted dice loss and a list for all class dice loss, expect background
        """
        dice_losses = []
        weight_dice_loss = 0
        all_slice = torch.split(input_tensor, [1] * self.num_class, dim=1)

        binary_dice_loss = BinaryDiceLoss()
        for i in range(self.num_class):
            # prepare for calculate label i dice loss
            slice_i = torch.cat([1 - all_slice[i], all_slice[i]], dim=1)
            target_i = (target == i).float()

            dice_i_loss = binary_dice_loss(slice_i, target_i)
            # save all classes dice loss and calculate weighted dice
            dice_losses.append(dice_i_loss)
            weight_dice_loss += dice_i_loss * self.weights[i]

        return weight_dice_loss