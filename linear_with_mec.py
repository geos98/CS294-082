import torch
import torch.nn as nn

def create_linear_layer(hidden_size = 18, in_feature = 64, out_feature = 10):
    net = nn.Sequential(
        nn.Linear(in_feature, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_feature)
    )

    mec = (in_feature + 1) * hidden_size + hidden_size

    return net, mec