import torch.nn as nn

def batch_norm(num_features):
    return nn.BatchNorm2d(num_features)

def layer_norm(num_features):
    return nn.LayerNorm(num_features)
