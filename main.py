import math
import os
import re

import matplotlib.pyplot
import numpy
import scipy.optimize
import torch
import torch.nn
import torch.nn.functional
import torch.nn.init
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal, Normal
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    Parameter,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import wandb
from wandb import Image


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


def evaluate(trainer):
    components, sample = generate_test_sample(1)

    with torch.no_grad():
        (
            predicted_existence,
            predicted_means,
            predicted_logvars,
        ) = trainer.condition(
            trainer.transform(sample.unsqueeze(0)),
            argmax=True,
        )

    num_components = int(predicted_existence.squeeze(0).sum().item())

    prediction = {
        "num_components": num_components,
        "means": predicted_means.squeeze(0)[:num_components],
        "logvars": predicted_logvars.squeeze(0)[:num_components],
    }

    gaussian_mixture = GaussianMixture(components, covariance_type="diag")

    gaussian_mixture.fit(sample)

    matplotlib.pyplot.scatter(
        sample[:, 0],
        sample[:, 1],
        c="k",
        label="Input Set",
        s=5,
        alpha=0.3,
    )

    for k in range(prediction["num_components"]):
        standard_deviation = numpy.sqrt(numpy.exp(prediction["logvars"][k]))

        elipse = Ellipse(
            prediction["means"][k],
            2 * standard_deviation[0].item(),
            2 * standard_deviation[1].item(),
            fill=False,
            edgecolor="r",
            linestyle="--",
            linewidth=2.5,
        )

        matplotlib.pyplot.gca().add_artist(elipse)

    for k in range(components):
        standard_deviation = numpy.sqrt(gaussian_mixture.covariances_[k])

        matplotlib.pyplot.gca().add_artist(
            Ellipse(
                gaussian_mixture.means_[k],
                2 * standard_deviation[0],
                2 * standard_deviation[1],
                fill=False,
                edgecolor="b",
                linestyle="--",
                linewidth=2.5,
            ),
        )

    matplotlib.pyplot.legend(
        [
            Line2D(
                [0],
                [0],
                color="r",
                linestyle="--",
                linewidth=1.5,
            ),
            Line2D(
                [0],
                [0],
                color="b",
                linestyle="--",
                linewidth=1.5,
            ),
        ],
        [
            "Amortized Mixture of Gaussians",
            "Gaussian Mixture",
        ],
    )

    wandb.log({f"1": Image(matplotlib.pyplot)})


def generate_test_sample(index: int) -> (int, Tensor):
    if index == 1:
        sample = torch.randn(100, 2) / 3.0
        sample[:, 0] = sample[:, 0] - 3.0
        sample = torch.cat([sample, torch.randn(100, 2) / 3.0 - 3.0], dim=0)
        sample = torch.cat([sample, torch.randn(100, 2) / 3.0 + 3.0], dim=0)
        components = 3
    elif index == 2:
        sample = torch.rand(100, 2)
        sample[:, 0] = sample[:, 0] * 5.0
        sample[:, 1] = sample[:, 1] + 2.0
        sample = torch.cat([sample, torch.randn(100, 2) / 3.0], dim=0)
        components = 2
    elif index == 3:
        sample = torch.randn(100, 2)
        sample[:, 0] = sample[:, 0] * 5.0
        input_set2 = torch.randn(100, 2)
        input_set2[:, 1] = input_set2[:, 1] * 5.0
        sample = torch.cat([sample, input_set2], dim=0)
        components = 2
    else:
        raise ValueError("Invalid case number")

    return components, sample


def reshape_last_dim(input, i, j):
    original_shape = input.shape
    last_dim_product = original_shape[-1]
    if i == -1 and j == -1:
        raise ValueError
    if i == -1:
        if last_dim_product % j != 0:
            raise ValueError
        i = last_dim_product // j
    elif j == -1:
        if last_dim_product % i != 0:
            raise ValueError
        j = last_dim_product // i
    elif i * j != last_dim_product:
        raise ValueError
    new_shape = original_shape[:-1] + (i, j)
    return torch.reshape(input, new_shape)


def generate_gaussian_mixture(
    batch_size,
    min_components,
    max_components,
    dim_output,
    min_dist,
    min_logvar,
    max_logvar,
    num_samples,
):
    components = torch.randint(min_components, max_components + 1, (batch_size,))

    means = torch.zeros(batch_size, max_components, dim_output)

    log_variances = torch.zeros(batch_size, max_components, dim_output)

    existences = torch.zeros(batch_size, max_components)

    for i in range(batch_size):
        k = components[i].item()
        current_means = []
        for j in range(k):
            while True:
                mean = torch.randn(dim_output) * 5
                valid = True

                for existing_mean in current_means:
                    if torch.norm(mean - existing_mean) < min_dist:
                        valid = False
                        break

                if valid:
                    current_means.append(mean)
                    means[i, j] = mean
                    log_variances[i, j] = (
                        torch.rand(dim_output) * (max_logvar - min_logvar) + min_logvar
                    )
                    existences[i, j] = 1
                    break

    samples = []

    for i in range(batch_size):
        k = components[i].item()
        mog_samples = []

        for j in range(k):
            mog_samples = [
                *mog_samples,
                MultivariateNormal(
                    means[i, j],
                    covariance_matrix=torch.diag(torch.exp(log_variances[i, j])),
                ).sample(
                    [
                        num_samples // k,
                    ],
                ),
            ]

        mog_samples = torch.concatenate(
            mog_samples,
            dim=0,
        )

        if num_samples % k != 0:
            mog_samples = torch.concatenate(
                [
                    mog_samples,
                    MultivariateNormal(
                        means[i, 0],
                        covariance_matrix=torch.diag(
                            torch.exp(log_variances[i, 0]),
                        ),
                    ).sample(
                        [
                            num_samples - mog_samples.shape[0],
                        ],
                    ),
                ],
                dim=0,
            )

        samples.append(mog_samples)

    samples = torch.stack(samples)

    parameters = {
        "existence": existences,
        "logvars": log_variances,
        "means": means,
        "num_components": components,
    }

    return parameters, samples


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz))
    mask = mask == 1
    mask = mask.transpose(0, 1)
    mask = mask.to(dtype=torch.get_default_dtype())
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def create_fixed_pos_encoding(max_len, d_model):
    output = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    x = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    x = torch.exp(x)
    output[:, 0::2] = torch.sin(position * x)
    output[:, 1::2] = torch.cos(position * x)
    return torch.unsqueeze(output, dim=0)


def calculate_metrics(
    predicted_existences: Tensor,
    predicted_means: Tensor,
    means,
    components,
) -> (Tensor, Tensor):
    predicted_components = torch.sum(
        predicted_existences,
        dim=1,
    )

    mean_distance = 0

    for i in range(predicted_means.shape[0]):
        pred_means_i = predicted_means[i, : int(predicted_components[i].item()), :]

        true_means_i = means[i, : components[i], :]

        if pred_means_i.shape[0] == 0 or true_means_i.shape[0] == 0:
            continue

        distances = torch.cdist(pred_means_i, true_means_i)

        distances = distances.detach().cpu().numpy()

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(distances)

        mean_distance = mean_distance + distances[row_ind, col_ind].mean()

    if predicted_means.shape[0] > 0:
        mean_distance = mean_distance / predicted_means.shape[0]
    else:
        mean_distance = torch.tensor(0.0)

    return mean_distance


def component_difference_score(prediction: Tensor, target: Tensor) -> Tensor:
    score = torch.sum(prediction, dim=1) == target

    score = score.to(dtype=torch.get_default_dtype())

    return torch.mean(score)


def loss(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    output = 0

    m = target[0].shape[0]
    n = target[0].shape[1]

    for i in range(m):
        for j in range(n):
            if mask[i, j] == 0:
                continue

            x = input[i, j, :]

            if len(x.shape) == 1:
                x = torch.unsqueeze(x, dim=0)

            output = torch.add(
                output,
                torch.squeeze(
                    torch.negative(
                        torch.sum(
                            torch.logsumexp(
                                torch.add(
                                    torch.unsqueeze(
                                        torch.unsqueeze(
                                            torch.log(target[0][i, j]),
                                            dim=0,
                                        ),
                                        dim=-1,
                                    ),
                                    Normal(
                                        target[1][i, j],
                                        target[2][i, j],
                                    ).log_prob(
                                        torch.unsqueeze(
                                            x,
                                            dim=1,
                                        ),
                                    ),
                                ),
                                dim=1,
                            ),
                            dim=-1,
                        ),
                    ),
                ),
            )

    if m > 0:
        output = output / m

    return output


def mask_loss(input: Tensor, target: Tensor) -> Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        torch.squeeze(
            target[:, : input.shape[1], :],
            dim=-1,
        ),
        input,
    )


class MixtureDensityNetwork(Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        components: int,
    ):
        super().__init__()

        self.components = components

        self.fc_pi = Linear(input_dimension, components)

        self.fc_mu = Linear(input_dimension, components * output_dimension)

        self.fc_sigma = Linear(input_dimension, components * output_dimension)

    def forward(self, input: Tensor) -> (Tensor, Tensor, Tensor):
        mixing_coefficients = torch.nn.functional.softmax(self.fc_pi(input), -1)

        means = reshape_last_dim(self.fc_mu(input), self.components, -1)

        variances = reshape_last_dim(
            torch.exp(self.fc_sigma(input)), self.components, -1
        )

        return mixing_coefficients, means, variances

    def sample(self, input: Tensor, argmax: bool = False) -> Tensor:
        mixing_coefficients, means, sigma = self.forward(input)

        if argmax:
            pis = torch.argmax(mixing_coefficients, -1).unsqueeze(-1).unsqueeze(-1)
        else:
            pis = Categorical(mixing_coefficients).sample().unsqueeze(-1).unsqueeze(-1)

        samples = torch.gather(means, dim=-2, index=pis.repeat(1, 1, 1, means.size(-1)))

        if not argmax:
            samples = samples + torch.randn_like(samples) * torch.gather(
                sigma, dim=-2, index=pis.repeat(1, 1, 1, sigma.size(-1))
            )

        return torch.squeeze(samples, dim=-2)


class MAB(Module):
    def __init__(
        self,
        dim_Q,
        dim_K,
        dim_V,
        num_heads,
        layer_normalization=False,
    ):
        super().__init__()

        self.dim_V = dim_V

        self.num_heads = num_heads

        self.fc_q = Linear(dim_Q, dim_V)
        self.fc_k = Linear(dim_K, dim_V)
        self.fc_v = Linear(dim_K, dim_V)

        if layer_normalization:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads

        q_ = torch.concatenate(Q.split(dim_split, 2), 0)
        k_ = torch.concatenate(K.split(dim_split, 2), 0)
        v_ = torch.concatenate(V.split(dim_split, 2), 0)

        attention = torch.softmax(q_.bmm(k_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)

        output = torch.concatenate(attention.bmm(v_).split(Q.size(0), 0), 2) + Q

        if getattr(self, "ln0", None) is None:
            output = output
        else:
            output = self.ln0(output)

        output = output + torch.nn.functional.relu(self.fc_o(output))

        if getattr(self, "ln1", None) is None:
            output = output
        else:
            output = self.ln1(output)

        return output


class SAB(Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        num_heads,
        layer_normalization=False,
    ):
        super().__init__()

        self.attention = MAB(
            input_dimension,
            input_dimension,
            output_dimension,
            num_heads,
            layer_normalization=layer_normalization,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.attention(input, input)


class ISAB(Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        num_heads,
        num_inds,
        layer_normalization=False,
    ):
        super().__init__()

        self.inducing_points = Parameter(torch.Tensor(1, num_inds, output_dimension))

        torch.nn.init.xavier_uniform_(self.inducing_points)

        self.attention_0 = MAB(
            output_dimension,
            input_dimension,
            output_dimension,
            num_heads,
            layer_normalization=layer_normalization,
        )

        self.attention_1 = MAB(
            input_dimension,
            output_dimension,
            output_dimension,
            num_heads,
            layer_normalization=layer_normalization,
        )

    def forward(self, X):
        return self.attention_1(
            X,
            self.attention_0(
                self.inducing_points.repeat(X.size(0), 1, 1),
                X,
            ),
        )


class PMA(Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        seeds: int = 1,
        layer_normalization=False,
    ):
        super().__init__()

        self.seeds = Parameter(torch.Tensor(1, seeds, dimension))

        torch.nn.init.xavier_uniform_(self.seeds)

        self.attention = MAB(
            dimension,
            dimension,
            dimension,
            num_heads,
            layer_normalization=layer_normalization,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.attention(self.seeds.repeat(input.size(0), 1, 1), input)


class SetTransformer(Module):
    def __init__(
        self,
        dim_input,
        dim_hidden,
        num_heads,
        layer_normalization: bool = False,
    ):
        super().__init__()

        self.encode = Sequential(
            SAB(
                dim_input,
                dim_hidden,
                num_heads,
                layer_normalization=layer_normalization,
            ),
            SAB(
                dim_hidden,
                dim_hidden,
                num_heads,
                layer_normalization=layer_normalization,
            ),
            SAB(
                dim_hidden,
                dim_hidden,
                num_heads,
                layer_normalization=layer_normalization,
            ),
            SAB(
                dim_hidden,
                dim_hidden,
                num_heads,
                layer_normalization=layer_normalization,
            ),
        )

        self.decode = Sequential(
            PMA(
                dim_hidden,
                num_heads,
                layer_normalization=layer_normalization,
            ),
            SAB(
                dim_hidden,
                dim_hidden,
                num_heads,
                layer_normalization=layer_normalization,
            ),
            SAB(
                dim_hidden,
                dim_hidden,
                num_heads,
                layer_normalization=layer_normalization,
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        return torch.squeeze(self.decode(self.encode(input)), dim=-2)


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
            dim_hidden, dim_output, components
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

            mask = generate_square_subsequent_mask(inputs.shape[1])

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
                mask = generate_square_subsequent_mask(output.shape[1])

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


class AmortizedMixtureOfGaussians(LightningModule):
    def __init__(
        self,
        dim_output,
        dim_hidden,
        num_heads,
        num_blocks,
        mdn_components,
        max_components,
        min_components,
        min_dist,
        min_logvar,
        max_logvar,
        num_samples,
        dropout=0.1,
        lr=1e-3,
        check_test_loss_every_n_epoch=1,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.transform = SetTransformer(
            dim_output,
            dim_hidden,
            num_heads,
            layer_normalization=False,
        )

        self.condition = ConditionalTransformer(
            dim_set_output=dim_hidden,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_blocks=num_blocks,
            maximum_components=max_components,
            components=mdn_components,
            dropout=dropout,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.condition(self.transform(input))

    def training_step(
        self,
        batch: (Tensor, Tensor, Tensor, Tensor, Tensor),
        batch_index: int,
    ) -> Tensor:
        (
            components,
            means,
            log_variances,
            masks,
            samples,
        ) = batch

        (
            predicted_means,
            predicted_log_variances,
            predicted_masks,
        ) = self.condition(
            self.transform(samples),
            torch.concatenate(
                [
                    torch.unsqueeze(
                        masks,
                        dim=-1,
                    ),
                    means,
                    log_variances,
                ],
                dim=-1,
            ),
        )

        train_mask_loss = mask_loss(
            masks,
            predicted_masks,
        )

        train_mean_loss = loss(
            means,
            predicted_means,
            mask=masks,
        )

        train_log_variance_loss = loss(
            log_variances,
            predicted_log_variances,
            mask=masks,
        )

        train_loss = train_mask_loss + train_mean_loss + train_log_variance_loss

        self.log(
            "train_loss",
            train_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        self.log(
            "train_mask_loss",
            train_mask_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
        )

        self.log(
            "train_mean_loss",
            train_mean_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
        )

        self.log(
            "train_log_variance_loss",
            train_log_variance_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
        )

        return train_loss

    def validation_step(
        self,
        batch: (Tensor, Tensor, Tensor, Tensor, Tensor),
        batch_index: int,
    ) -> Tensor:
        (
            components,
            means,
            log_variances,
            masks,
            samples,
        ) = batch

        (
            predicted_means,
            predicted_log_variances,
            predicted_masks,
        ) = self.condition(
            self.transform(samples),
            torch.concatenate(
                [
                    torch.unsqueeze(
                        masks,
                        dim=-1,
                    ),
                    means,
                    log_variances,
                ],
                dim=-1,
            ),
        )

        val_mask_loss = mask_loss(
            masks,
            predicted_masks,
        )

        val_mean_loss = loss(
            means,
            predicted_means,
            mask=masks,
        )

        val_log_variance_loss = loss(
            log_variances,
            predicted_log_variances,
            mask=masks,
        )

        val_loss = val_mask_loss + val_mean_loss + val_log_variance_loss

        self.log(
            "val_loss",
            val_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        self.log(
            "val_mask_loss",
            val_mask_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
        )

        self.log(
            "val_mean_loss",
            val_mean_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
        )

        self.log(
            "val_log_variance_loss",
            val_log_variance_loss,
            logger=True,
            on_epoch=True,
            on_step=True,
        )

        return val_loss

    def test_step(
        self,
        batch: (Tensor, Tensor, Tensor, Tensor, Tensor),
        batch_index: int,
    ) -> (Tensor, Tensor):
        (
            components,
            means,
            log_variances,
            masks,
            samples,
        ) = batch

        (
            predicted_means,
            predicted_log_variances,
            predicted_masks,
        ) = self.condition(self.transform(samples))

        test_component_differences = component_difference_score(
            predicted_masks,
            components,
        )

        test_mean_distances = calculate_metrics(
            predicted_masks,
            predicted_means,
            means,
            components,
        )

        self.log(
            "test_component_differences",
            test_component_differences,
        )

        self.log(
            "test_mean_distances",
            test_mean_distances,
        )

        return test_component_differences, test_mean_distances

    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.check_test_loss_every_n_epoch == 0:
            self.evaluate_test_metrics()

        self.prepare_data(train_only=True)

    def evaluate_test_metrics(self):
        self.eval()

        component_differences = []

        mean_distances = []

        for batch in self.test_dataloader():
            (
                components,
                means,
                log_variances,
                masks,
                samples,
            ) = batch

            (
                predicted_means,
                predicted_log_variances,
                predicted_masks,
            ) = self.condition(self.transform(samples))

            component_difference = component_difference_score(
                predicted_masks,
                components,
            )

            mean_distance = calculate_metrics(
                predicted_masks,
                predicted_means,
                means,
                components,
            )

            component_differences = [
                *component_differences,
                component_difference,
            ]

            mean_distances = [
                *mean_distances,
                mean_distance,
            ]

        evaluate(self)

        self.train()

        test_periodic_component_difference = torch.mean(
            torch.stack(component_differences),
        )

        if mean_distances:
            test_periodic_mean_distance = torch.mean(
                torch.tensor(mean_distances),
            )
        else:
            test_periodic_mean_distance = torch.tensor(0.0)

        self.log(
            "test_periodic_component_difference",
            test_periodic_component_difference,
        )

        self.log(
            "test_periodic_mean_distance",
            test_periodic_mean_distance,
        )

    def configure_optimizers(self) -> Adam:
        return Adam(self.parameters(), lr=self.hparams.lr)

    def prepare_data(self, train_only=False):
        train_parameters, train_samples = generate_gaussian_mixture(
            batch_size=10000,
            min_components=self.hparams.min_components,
            max_components=self.hparams.max_components,
            dim_output=self.hparams.dim_output,
            min_dist=self.hparams.min_dist,
            min_logvar=self.hparams.min_logvar,
            max_logvar=self.hparams.max_logvar,
            num_samples=self.hparams.num_samples,
        )

        train_parameters = (
            train_parameters["num_components"],
            train_parameters["means"],
            train_parameters["logvars"],
            train_parameters["existence"],
        )

        self.train_dataset = TensorDataset(*train_parameters, train_samples)

        if not train_only:
            val_parameters, val_samples = generate_gaussian_mixture(
                batch_size=500,
                min_components=self.hparams.min_components,
                max_components=self.hparams.max_components,
                dim_output=self.hparams.dim_output,
                min_dist=self.hparams.min_dist,
                min_logvar=self.hparams.min_logvar,
                max_logvar=self.hparams.max_logvar,
                num_samples=self.hparams.num_samples,
            )

            val_parameters = (
                val_parameters["num_components"],
                val_parameters["means"],
                val_parameters["logvars"],
                val_parameters["existence"],
            )

            self.val_dataset = TensorDataset(*val_parameters, val_samples)

            test_parameters, test_samples = generate_gaussian_mixture(
                batch_size=500,
                min_components=self.hparams.min_components,
                max_components=self.hparams.max_components,
                dim_output=self.hparams.dim_output,
                min_dist=self.hparams.min_dist,
                min_logvar=self.hparams.min_logvar,
                max_logvar=self.hparams.max_logvar,
                num_samples=self.hparams.num_samples,
            )

            test_parameters = (
                test_parameters["num_components"],
                test_parameters["means"],
                test_parameters["logvars"],
                test_parameters["existence"],
            )

            self.test_dataset = TensorDataset(*test_parameters, test_samples)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=32,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=32,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=32,
        )


if __name__ == "__main__":
    trainer = Trainer(
        accelerator="cpu",
        callbacks=[
            EarlyStopping(
                mode="min",
                monitor="val_loss",
                patience=32,
                verbose=True,
            ),
        ],
        logger=WandbLogger(
            project="amortized-mixture-of-gaussians",
            log_model=True,
        ),
        max_epochs=256,
    )

    model = AmortizedMixtureOfGaussians(
        dim_output=2,
        dim_hidden=64,
        num_heads=4,
        num_blocks=6,
        mdn_components=3,
        dropout=0.1,
        max_components=5,
        min_components=1,
        min_dist=2.0,
        min_logvar=-2.0,
        max_logvar=2.0,
        num_samples=100,
        check_test_loss_every_n_epoch=1,
        lr=1e-3,
    )

    trainer.fit(model)
    trainer.test(model)

    wandb.finish()
