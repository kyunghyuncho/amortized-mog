import os
import re

import torch

# this function finds the latest checkpoint from `checkpoint_path`.
# checkpoint files are in the format of `epoch={epoch}-step={step}.ckpt`.
def find_latest_checkpoint(checkpoint_path):
    def extract_epoch_and_step(filename):
        m = re.match(r"epoch=(\d+)-step=(\d+).ckpt", filename)
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2))

    def compare_epoch_and_step(a, b):
        assert a is not None
        if b is None:
            return True
        return a[0] > b[0] or (a[0] == b[0] and a[1] > b[1])

    latest_epoch_and_step = None
    latest_checkpoint = None
    for filename in os.listdir(checkpoint_path):
        epoch_and_step = extract_epoch_and_step(filename)
        if compare_epoch_and_step(epoch_and_step, latest_epoch_and_step):
            latest_epoch_and_step = epoch_and_step
            latest_checkpoint = filename
    return f"{checkpoint_path}/{latest_checkpoint}"

# compute the log probability of a set of samples under a mixture of Gaussians
def log_prob_mog(samples, pi, mu, sigma):
    if len(samples.shape) == 1:
        samples = samples.unsqueeze(0)
    dist = torch.distributions.Normal(mu, sigma)
    log_lik = dist.log_prob(samples.unsqueeze(1))
    log_probs = torch.logsumexp(torch.log(pi).unsqueeze(0).unsqueeze(-1) + log_lik, 1)
    return torch.sum(log_probs, -1)

def reshape_last_dim(tensor, i, j):
    """Reshapes the last dimension of a PyTorch tensor into (i, j),
    supporting -1 for automatic calculation of one of the dimensions.

    Args:
        tensor: The input tensor with arbitrary number of dimensions.
        i: The first dimension of the reshaped last dimension. Can be -1.
        j: The second dimension of the reshaped last dimension. Can be -1.
        
    Returns:
        A new tensor with the last dimension reshaped to (i, j).
    """

    original_shape = tensor.shape
    last_dim_product = original_shape[-1]

    if i == -1 and j == -1:
        raise ValueError("At most one of i and j can be -1")

    if i == -1:
        if last_dim_product % j != 0:
            raise ValueError(f"Last dimension size {last_dim_product} is not divisible by {j} when i is -1")
        i = last_dim_product // j
    elif j == -1:
        if last_dim_product % i != 0:
            raise ValueError(f"Last dimension size {last_dim_product} is not divisible by {i} when j is -1")
        j = last_dim_product // i
    elif i * j != last_dim_product:
        raise ValueError(f"Cannot reshape last dimension of size {last_dim_product} into ({i}, {j})")

    new_shape = original_shape[:-1] + (i, j)
    reshaped_tensor = tensor.reshape(new_shape)

    return reshaped_tensor