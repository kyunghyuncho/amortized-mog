import math

import torch


def create_fixed_pos_encoding(max_len, d_model):
    output = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    x = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    x = torch.exp(x)
    output[:, 0::2] = torch.sin(position * x)
    output[:, 1::2] = torch.cos(position * x)
    return torch.unsqueeze(output, dim=0)
