import torch.nn as nn
import torch.nn.functional as F


class EntropyMinimizationLoss(nn.Module):
    def __init__(self):
        super(EntropyMinimizationLoss, self).__init__()

    def forward(self, q, p):
        res = F.softmax(q, dim=1) * F.log_softmax(p, dim=1)
        res = -1.0 * res.mean()
        return res