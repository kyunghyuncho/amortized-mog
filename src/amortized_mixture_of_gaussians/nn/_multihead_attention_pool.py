import torch
from torch import Tensor
from torch.nn import Module, Parameter

from ._multihead_attention_block import MultiHeadAttentionBlock


class MultiHeadAttentionPool(Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        seeds: int = 1,
        normalize=False,
    ):
        super().__init__()

        self.seeds = Parameter(torch.Tensor(1, seeds, dimension))

        torch.nn.init.xavier_uniform_(self.seeds)

        self.attention = MultiHeadAttentionBlock(
            dimension,
            dimension,
            dimension,
            num_heads,
            normalize=normalize,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.attention(self.seeds.repeat(input.size(0), 1, 1), input)
