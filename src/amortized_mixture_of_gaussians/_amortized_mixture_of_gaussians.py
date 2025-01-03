import matplotlib.pyplot
import numpy
import torch
import wandb
from lightning import LightningModule
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from wandb import Image

from src.amortized_mixture_of_gaussians.datasets._generate_gaussian_mixture import generate_gaussian_mixture
from .datasets import generate_test_sample
from .metrics.functional import (
    component_difference_score,
    mean_distance_score,
)
from .nn import (
    ConditionalTransformer,
    SetTransformer,
)
from .nn.functional import (
    amortized_mixture_of_gaussians_loss,
    amortized_mixture_of_gaussians_mask_loss,
)


class AmortizedMixtureOfGaussians(LightningModule):
    def __init__(
        self,
        dim_output,
        dim_hidden,
        num_heads,
        num_blocks,
        components,
        maximum_components,
        minimum_components,
        minimum_distance,
        minimum_log_variance,
        maximum_log_variance,
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
            normalize=False,
        )

        self.condition = ConditionalTransformer(
            dim_set_output=dim_hidden,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_blocks=num_blocks,
            maximum_components=maximum_components,
            components=components,
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

        train_mask_loss = amortized_mixture_of_gaussians_mask_loss(
            masks,
            predicted_masks,
        )

        train_mean_loss = amortized_mixture_of_gaussians_loss(
            means,
            predicted_means,
            mask=masks,
        )

        train_log_variance_loss = amortized_mixture_of_gaussians_loss(
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

        val_mask_loss = amortized_mixture_of_gaussians_mask_loss(
            masks,
            predicted_masks,
        )

        val_mean_loss = amortized_mixture_of_gaussians_loss(
            means,
            predicted_means,
            mask=masks,
        )

        val_log_variance_loss = amortized_mixture_of_gaussians_loss(
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

        test_component_difference = component_difference_score(
            predicted_masks,
            components,
        )

        test_mean_distance = mean_distance_score(
            predicted_masks,
            predicted_means,
            means,
            components,
        )

        self.log(
            "test_component_difference",
            test_component_difference,
        )

        self.log(
            "test_mean_distance",
            test_mean_distance,
        )

        return test_component_difference, test_mean_distance

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

            mean_distance = mean_distance_score(
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

        for index in range(3):
            self.plot(index)

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
            minimum_components=self.hparams.minimum_components,
            maximum_components=self.hparams.maximum_components,
            dim_output=self.hparams.dim_output,
            minimum_distance=self.hparams.minimum_distance,
            minimum_log_variance=self.hparams.minimum_log_variance,
            maximum_log_variance=self.hparams.maximum_log_variance,
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
                minimum_components=self.hparams.minimum_components,
                maximum_components=self.hparams.maximum_components,
                dim_output=self.hparams.dim_output,
                minimum_distance=self.hparams.minimum_distance,
                minimum_log_variance=self.hparams.minimum_log_variance,
                maximum_log_variance=self.hparams.maximum_log_variance,
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
                minimum_components=self.hparams.minimum_components,
                maximum_components=self.hparams.maximum_components,
                dim_output=self.hparams.dim_output,
                minimum_distance=self.hparams.minimum_distance,
                minimum_log_variance=self.hparams.minimum_log_variance,
                maximum_log_variance=self.hparams.maximum_log_variance,
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

    def plot(self, index):
        components, sample = generate_test_sample(index + 1)

        with torch.no_grad():
            (
                predicted_means,
                predicted_log_variances,
                predicted_masks,
            ) = self.condition(
                self.transform(
                    torch.unsqueeze(
                        sample,
                        dim=0,
                    ),
                ),
                argmax=True,
            )

        num_components = int(predicted_masks.squeeze(0).sum().item())

        prediction = {
            "num_components": num_components,
            "means": predicted_means.squeeze(0)[:num_components],
            "logvars": predicted_log_variances.squeeze(0)[:num_components],
        }

        gaussian_mixture = GaussianMixture(components, covariance_type="diag")

        gaussian_mixture.fit(sample)

        figure, axes = matplotlib.pyplot.subplots()

        axes.scatter(
            sample[:, 0],
            sample[:, 1],
            s=5,
            c="k",
            alpha=0.3,
        )

        for k in range(prediction["num_components"]):
            standard_deviation = numpy.sqrt(numpy.exp(prediction["logvars"][k]))

            figure.gca().add_artist(
                Ellipse(
                    prediction["means"][k],
                    2 * standard_deviation[0].item(),
                    2 * standard_deviation[1].item(),
                    fill=False,
                    edgecolor="r",
                    linestyle="--",
                    linewidth=2.5,
                ),
            )

        for k in range(components):
            standard_deviation = numpy.sqrt(gaussian_mixture.covariances_[k])

            figure.gca().add_artist(
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

        figure.legend(
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
                "Expectation-Maximization ",
            ],
        )

        wandb.log({f"{index}": Image(figure)})
