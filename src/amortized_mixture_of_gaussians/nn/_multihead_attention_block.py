import math

import torch
from torch import Tensor
from torch.nn import Module, Linear, LayerNorm


class MultiHeadAttentionBlock(Module):
    def __init__(
            self,
            q_dimension: int,
            k_dimension: int,
            v_dimension: int,
            num_heads: int,
            normalize: bool = False,
    ):
        super().__init__()

        self.v_dimension = v_dimension

        self.num_heads = num_heads

        self.q = Linear(q_dimension, v_dimension)
        self.k = Linear(k_dimension, v_dimension)
        self.v = Linear(k_dimension, v_dimension)

        if normalize:
            self.layer_normalization_0 = LayerNorm(v_dimension)
            self.layer_normalization_1 = LayerNorm(v_dimension)

        self.output = Linear(v_dimension, v_dimension)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        q = self.q(query)
        k = self.k(key)
        v = self.v(key)

        dim_split = self.v_dimension // self.num_heads

        q__ = torch.concatenate(q.split(dim_split, 2), dim=0)
        k__ = torch.concatenate(k.split(dim_split, 2), dim=0)
        v__ = torch.concatenate(v.split(dim_split, 2), dim=0)

        attention = q__.bmm(k__.transpose(1, 2))

        attention = attention / math.sqrt(dim_split)

        attention = torch.softmax(
            attention,
            dim=2,
        )

        output = (q__ + attention.bmm(v__))

        output = output.split(q.size(0), 0)

        output = torch.concatenate(
            output,
            dim=2,
        )

        if getattr(self, "layer_normalization_0", None) is not None:
            output = self.layer_normalization_0(output)

        output = output + torch.nn.functional.relu(self.output(output))

        if getattr(self, "layer_normalization_1", None) is not None:
            output = self.layer_normalization_1(output)

        return output
