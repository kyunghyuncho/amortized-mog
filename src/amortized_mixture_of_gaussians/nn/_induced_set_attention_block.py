import torch
from torch.nn import Module, Parameter

from ._multihead_attention_block import MultiHeadAttentionBlock


class InducedSetAttentionBlock(Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        num_heads,
        num_inds,
        normalize=False,
    ):
        super().__init__()

        self.inducing_points = Parameter(torch.Tensor(1, num_inds, output_dimension))

        torch.nn.init.xavier_uniform_(self.inducing_points)

        self.attention_0 = MultiHeadAttentionBlock(
            output_dimension,
            input_dimension,
            output_dimension,
            num_heads,
            normalize=normalize,
        )

        self.attention_1 = MultiHeadAttentionBlock(
            input_dimension,
            output_dimension,
            output_dimension,
            num_heads,
            normalize=normalize,
        )

    def forward(self, X):
        return self.attention_1(
            X,
            self.attention_0(
                self.inducing_points.repeat(X.size(0), 1, 1),
                X,
            ),
        )
