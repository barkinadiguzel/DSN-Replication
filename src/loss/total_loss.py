import torch

def squared_hinge_loss(logits, target):
    target_onehot = torch.nn.functional.one_hot(target, num_classes=logits.size(1)).float()
    margins = 1 - logits * (2*target_onehot - 1)
    loss = torch.mean(torch.clamp(margins, min=0)**2)
    return loss

def total_loss(output, hidden_outputs, target, alpha_list, gamma):
    loss_out = squared_hinge_loss(output, target)
    loss_comp = 0
    for alpha, h_out in zip(alpha_list, hidden_outputs):
        comp_loss = squared_hinge_loss(h_out, target)
        comp_loss = torch.clamp(comp_loss - gamma, min=0)
        loss_comp += alpha * comp_loss
    return loss_out + loss_comp
