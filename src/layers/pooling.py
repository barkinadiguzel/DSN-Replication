import torch.nn as nn

def max_pool(kernel_size=2, stride=2):
    return nn.MaxPool2d(kernel_size, stride)

def avg_pool(kernel_size=2, stride=2):
    return nn.AvgPool2d(kernel_size, stride)
