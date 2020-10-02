import torch

from segmentation3d.loss.cross_entropy_loss import CrossEntropyLoss

torch.manual_seed(0)
fps = torch.rand(1, 2, 2, 2, 2)
fps.requires_grad = True

probs = torch.nn.Softmax(dim=1)(fps)
vals, labels = probs.max(dim=1)
print(vals.requires_grad, labels.requires_grad)

index = vals.ge(0.8)
vals_valid = vals[index]
labels_valid = labels[index]

loss_func_ce = CrossEntropyLoss()
loss_val_ce = loss_func_ce(probs, labels)
print(probs, vals, labels, loss_val_ce)
