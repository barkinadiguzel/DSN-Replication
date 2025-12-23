import torch
import torch.nn.functional as F

def squared_hinge_loss(logits, target):
    target_onehot = F.one_hot(target, num_classes=logits.size(1)).float()
    margins = 1 - logits * (2*target_onehot - 1)
    loss = torch.mean(torch.clamp(margins, min=0)**2)
    return loss
