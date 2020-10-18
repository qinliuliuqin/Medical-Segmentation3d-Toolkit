import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(self, weights=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        if weights is not None:
            weights = torch.FloatTensor(weights)
            weights = weights / weights.sum()

        self.func = nn.CrossEntropyLoss(weights, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target):
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        if target.dim() == input.dim() == 5 or target.dim() == input.dim() == 3:
            target = torch.squeeze(target, dim=1)

        return self.func(input, target.long())


if __name__ == '__main__':

    func = CrossEntropyLoss([0.1, 0.9])
    input = torch.Tensor([[[0.2], [0.8]], [[0.7], [0.3]]])
    target = torch.Tensor([[1], [0]]).long()
    print(input.shape, target.shape)
    print(func(input, target))

    import math
    val1 = -0.8 + math.log(math.exp(0.2) + math.exp(0.8))
    val2 = -0.7 + math.log(math.exp(0.3) + math.exp(0.7))
    val3 = (0.9*val1 + 0.1*val2) / 1.0
    print(val1, val2, val3)