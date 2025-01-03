import torch
from torch import Tensor
from torch.nn import (
    Linear,
    Module,
    Parameter,
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from ._mixture_density_network import MixtureDensityNetwork
from .functional._create_fixed_pos_encoding import create_fixed_pos_encoding


class ConditionalTransformer(Module):
    def __init__(
        self,
        dim_set_output,
        dim_output,
        dim_hidden,
        num_heads,
        num_blocks,
        maximum_components,
        components,
        dropout=0.1,
    ):
        super().__init__()

        self.maximum_components = maximum_components

        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.embedding = Linear(dim_set_output + 1 + 2 * dim_output, dim_hidden)

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=dim_hidden,
                nhead=num_heads,
                dim_feedforward=dim_hidden,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=num_blocks,
        )

        self.mask_predictor = Linear(dim_hidden, 1)

        self.mean_predictor = MixtureDensityNetwork(dim_hidden, dim_output, components)

        self.log_variance_predictor = MixtureDensityNetwork(
            dim_hidden,
            dim_output,
            components,
        )

        self.sos_token = Parameter(torch.randn(1, 1, 1 + 2 * dim_output))

        self.positional_encoding = create_fixed_pos_encoding(
            3 * maximum_components, dim_hidden
        )

    def forward(
        self,
        input: Tensor,
        targets: Tensor | None = None,
        argmax: bool = True,
    ) -> (Tensor, Tensor, Tensor):
        batch_size = input.shape[0]

        tokens = self.sos_token.repeat(batch_size, 1, 1)

        if targets is not None:
            inputs = self.embedding(
                torch.concatenate(
                    [
                        torch.unsqueeze(
                            input,
                            dim=1,
                        ).repeat([1, self.maximum_components + 1, 1]),
                        torch.concatenate(
                            [
                                tokens,
                                targets,
                            ],
                            dim=1,
                        ),
                    ],
                    dim=-1,
                ),
            )

            inputs = inputs + self.positional_encoding[:, : inputs.shape[1], :]

            mask = Transformer.generate_square_subsequent_mask(inputs.shape[1])

            output = self.transformer(inputs, mask=mask)

            predicted_masks = self.mask_predictor(output[:, :-1, :])

            predicted_means = self.mean_predictor(output[:, :-1, :])

            predicted_log_variances = self.log_variance_predictor(output[:, :-1, :])
        else:
            (
                predicted_masks,
                predicted_means,
                predicted_log_variances,
            ) = None, None, None

            output = torch.add(
                self.embedding(
                    torch.concatenate(
                        [
                            torch.unsqueeze(
                                input,
                                dim=1,
                            ),
                            tokens,
                        ],
                        dim=-1,
                    ),
                ),
                self.positional_encoding[:, :1, :],
            )

            for _ in range(self.maximum_components):
                mask = Transformer.generate_square_subsequent_mask(output.shape[1])

                transformer_features = torch.unsqueeze(
                    self.transformer(
                        output,
                        mask=mask,
                    )[:, -1, :],
                    dim=1,
                )

                predicted_existence = self.mask_predictor(
                    transformer_features,
                )

                if argmax:
                    mask = (predicted_existence > 0).float()

                    mean = self.mean_predictor.sample(
                        transformer_features,
                        argmax=True,
                    )

                    log_variance = self.log_variance_predictor.sample(
                        transformer_features,
                        argmax=True,
                    )
                else:
                    mask = torch.sigmoid(predicted_existence) > torch.rand(1)

                    mask = mask.to(dtype=torch.get_default_dtype())

                    mean = self.mean_predictor.sample(
                        transformer_features,
                        argmax=False,
                    )

                    log_variance = self.log_variance_predictor.sample(
                        transformer_features,
                        argmax=False,
                    )

                if predicted_masks is None:
                    (
                        predicted_masks,
                        predicted_means,
                        predicted_log_variances,
                    ) = mask, mean, log_variance
                else:
                    predicted_masks = torch.concatenate(
                        [
                            predicted_masks,
                            mask,
                        ],
                        dim=1,
                    )

                    predicted_means = torch.concatenate(
                        [
                            predicted_means,
                            mean,
                        ],
                        dim=1,
                    )

                    predicted_log_variances = torch.concatenate(
                        [
                            predicted_log_variances,
                            log_variance,
                        ],
                        dim=1,
                    )

                output = torch.concatenate(
                    [
                        output,
                        torch.add(
                            self.embedding(
                                torch.concatenate(
                                    [
                                        torch.unsqueeze(
                                            input,
                                            dim=1,
                                        ),
                                        mask,
                                        mean,
                                        log_variance,
                                    ],
                                    dim=-1,
                                ),
                            ),
                            torch.unsqueeze(
                                self.positional_encoding[:, output.shape[1], :],
                                dim=1,
                            ),
                        ),
                    ],
                    dim=1,
                )

        return predicted_means, predicted_log_variances, predicted_masks
