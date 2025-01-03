from torch import Tensor
from torch.nn import Module

from ._multihead_attention_block import MultiHeadAttentionBlock


class SetAttentionBlock(Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        num_heads,
        normalize=False,
    ):
        super().__init__()

        self.attention = MultiHeadAttentionBlock(
            input_dimension,
            input_dimension,
            output_dimension,
            num_heads,
            normalize=normalize,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.attention(input, input)
