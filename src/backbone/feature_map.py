import torch

def flatten_feature_map(feature_map):
    return feature_map.view(feature_map.size(0), -1)
