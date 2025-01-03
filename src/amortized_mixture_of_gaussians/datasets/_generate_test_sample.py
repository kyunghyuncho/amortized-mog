import torch
from torch import Tensor


def generate_test_sample(index: int) -> (int, Tensor):
    if index == 1:
        components, sample = generate_test_sample_1()
    elif index == 2:
        components, sample = generate_test_sample_2()
    elif index == 3:
        components, sample = generate_test_sample_3()
    else:
        raise ValueError

    return components, sample

def generate_test_sample_1():
    x = torch.randn(100, 2) / 3.0
    y = torch.randn(100, 2) / 3.0 - 3.0
    z = torch.randn(100, 2) / 3.0 + 3.0

    x[:, 0] = x[:, 0] - 3.0

    return 3, torch.concatenate([torch.concatenate([x, y], dim=0), z], dim=0)


def generate_test_sample_2():
    x = torch.rand(100, 2)
    y = torch.randn(100, 2) / 3.0

    x[:, 0] = x[:, 0] * 5.0
    x[:, 1] = x[:, 1] + 2.0

    return 2, torch.concatenate([x, y], dim=0)


def generate_test_sample_3():
    x = torch.randn(100, 2)
    y = torch.randn(100, 2)

    x[:, 0] = x[:, 0] * 5.0
    y[:, 1] = y[:, 1] * 5.0

    return 2, torch.concatenate([x, y], dim=0)
