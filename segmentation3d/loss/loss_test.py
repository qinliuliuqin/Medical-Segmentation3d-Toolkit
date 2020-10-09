import torch

from segmentation3d.loss.cross_entropy_loss import CrossEntropyLoss
from segmentation3d.loss.entropy_minization import EntropyMinimizationLoss

torch.manual_seed(0)
fps = torch.rand(1, 2, 3, 3, 3)

val, mask = fps.max(dim=1)
idx = val > 0.5

mask_valid = mask[idx]
val_valid = fps[:, :, idx[0,:]]

print(mask_valid, val_valid)
print(mask_valid.shape, val_valid.shape)


# fps.requires_grad = True
# probs = torch.nn.Softmax(dim=1)(fps)
# vals, labels = probs.max(dim=1)
# print(vals.requires_grad, labels.requires_grad)
# index = vals.ge(0.8)
# vals_valid = vals[index]
# labels_valid = labels[index]
# loss_func_ce = CrossEntropyLoss()
# loss_val_ce = loss_func_ce(probs, labels)
# print(probs, vals, labels, loss_val_ce)
# fps_n = torch.rand_like(fps)
# print(fps_n)
# print(EntropyMinimizationLoss()(fps, fps))