import torch


def reshape_last_dim(input, i, j):
    if i == -1 and j == -1:
        raise ValueError

    if i == -1:
        if input.shape[-1] % j != 0:
            raise ValueError

        i = input.shape[-1] // j
    elif j == -1:
        if input.shape[-1] % i != 0:
            raise ValueError

        j = input.shape[-1] // i
    elif i * j != input.shape[-1]:
        raise ValueError

    return torch.reshape(input, input.shape[:-1] + (i, j))
