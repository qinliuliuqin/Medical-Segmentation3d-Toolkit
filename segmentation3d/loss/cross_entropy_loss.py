import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        self.func = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target):
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        target = torch.squeeze(target, dim=1).long()
        return self.func(input, target)


if __name__ == '__main__':

    func = CrossEntropyLoss()
    input = torch.Tensor([[[0.0], [1.0]]])
    target = torch.Tensor([[[1]]]).long()
    print(func(input, target))
