import torch
from torch import Tensor
from torch.nn import Module, Sequential

from ._multihead_attention_pool import MultiHeadAttentionPool
from ._set_attention_block import SetAttentionBlock


class SetTransformer(Module):
    def __init__(
        self,
        dim_input: int,
        d_model: int,
        nhead: int,
        normalize: bool = False,
    ):
        super().__init__()

        self.encode = Sequential(
            SetAttentionBlock(
                dim_input,
                d_model,
                nhead,
                normalize=normalize,
            ),
            SetAttentionBlock(
                d_model,
                d_model,
                nhead,
                normalize=normalize,
            ),
            SetAttentionBlock(
                d_model,
                d_model,
                nhead,
                normalize=normalize,
            ),
            SetAttentionBlock(
                d_model,
                d_model,
                nhead,
                normalize=normalize,
            ),
        )

        self.decode = Sequential(
            MultiHeadAttentionPool(
                d_model,
                nhead,
                normalize=normalize,
            ),
            SetAttentionBlock(
                d_model,
                d_model,
                nhead,
                normalize=normalize,
            ),
            SetAttentionBlock(
                d_model,
                d_model,
                nhead,
                normalize=normalize,
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        return torch.squeeze(self.decode(self.encode(input)), dim=-2)
