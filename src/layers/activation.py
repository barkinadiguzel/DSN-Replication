import torch.nn.functional as F

def relu(x):
    return F.relu(x)

def leaky_relu(x, negative_slope=0.01):
    return F.leaky_relu(x, negative_slope)

def sigmoid(x):
    return F.sigmoid(x)
