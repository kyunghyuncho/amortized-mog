import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from amortized_mog import ConditionalTransformerLM
from synthetic_mog import generate_gaussian_mixture
from modules import SetTransformer2
import wandb
from scipy.optimize import linear_sum_assignment
import numpy as np

class MoGTrainer(pl.LightningModule):
    def __init__(self, dim_input, dim_output, dim_hidden, num_heads, num_blocks, max_components,
                 min_components, min_dist, min_logvar, max_logvar, num_samples, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate SetTransformer++ and ConditionalTransformerLM
        self.set_transformer = SetTransformer2(dim_input, dim_hidden, num_heads, num_blocks, dim_output)
        self.conditional_lm = ConditionalTransformerLM(
            dim_set_output=dim_hidden,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_components=max_components,
            vocab_size=100 # not used in this case
        )

    def forward(self, x):
        # Pass input through SetTransformer++
        set_transformer_output = self.set_transformer(x)

        # Pass SetTransformer++ output through ConditionalTransformerLM
        existence_logits, means, logvars = self.conditional_lm(set_transformer_output)
        return existence_logits, means, logvars

    def training_step(self, batch, batch_idx):
        num_components, means, logvars, existence, samples = batch
        mog_params = {
            "num_components": num_components,
            "means": means,
            "logvars": logvars,
            "existence": existence
        }

        # Pass input through SetTransformer++
        set_transformer_output = self.set_transformer(samples)

        # Prepare targets for ConditionalTransformerLM
        targets = torch.cat([existence.unsqueeze(-1), means, logvars], dim=-1)

        # Pass SetTransformer++ output and targets through ConditionalTransformerLM
        existence_logits, pred_means, pred_logvars = self.conditional_lm(set_transformer_output, targets)

        loss = self.calculate_loss(existence_logits, pred_means, pred_logvars, mog_params)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        num_components, means, logvars, existence, samples = batch
        mog_params = {
            "num_components": num_components,
            "means": means,
            "logvars": logvars,
            "existence": existence
        }

        # Pass input through SetTransformer++
        set_transformer_output = self.set_transformer(samples)

        # Prepare targets for ConditionalTransformerLM
        targets = torch.cat([existence.unsqueeze(-1), means, logvars], dim=-1)

        # Pass SetTransformer++ output and targets through ConditionalTransformerLM
        existence_logits, pred_means, pred_logvars = self.conditional_lm(set_transformer_output, targets)

        loss = self.calculate_loss(existence_logits, pred_means, pred_logvars, mog_params)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        num_components, means, logvars, existence, samples = batch
        mog_params = {
            "num_components": num_components,
            "means": means,
            "logvars": logvars,
            "existence": existence
        }

        # Pass input through SetTransformer++
        set_transformer_output = self.set_transformer(samples)

        # Pass SetTransformer++ output through ConditionalTransformerLM for inference
        existence_logits, pred_means, pred_logvars = self.conditional_lm(set_transformer_output)

        # 1. Check accuracy of the number of components
        pred_num_components = (torch.sigmoid(existence_logits.squeeze(-1)) > 0.5).sum(dim=1)
        num_components_accuracy = (pred_num_components == mog_params["num_components"]).float().mean()
        self.log("test_num_components_accuracy", num_components_accuracy)

        # 2. Calculate mean distance after best matching
        total_mean_distance = 0
        for i in range(means.shape[0]):  # Iterate over the batch
            pred_means_i = pred_means[i, :pred_num_components[i], :]  # Get predicted means for existing components
            true_means_i = mog_params["means"][i, :mog_params["num_components"][i], :]  # Get true means

            if pred_means_i.shape[0] == 0 or true_means_i.shape[0] == 0:
                continue

            # Calculate pairwise distances
            distances = torch.cdist(pred_means_i, true_means_i)  # Pairwise Euclidean distances

            # Convert to numpy for linear_sum_assignment
            distances_np = distances.detach().cpu().numpy()

            # Find best matching using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(distances_np)

            # Calculate mean distance for the best matching
            mean_distance = distances_np[row_ind, col_ind].mean()
            total_mean_distance += mean_distance

        # Average mean distance over the batch
        if means.shape[0] > 0:
          avg_mean_distance = total_mean_distance / means.shape[0]
          self.log("test_mean_distance", avg_mean_distance)

        return num_components_accuracy, avg_mean_distance

    def calculate_loss(self, existence_logits, pred_means, pred_logvars, mog_params):
        # Get the maximum number of components
        max_components = mog_params["existence"].shape[1]

        # Slice the existence_logits tensor to match the size of mog_params["existence"]
        existence_logits = existence_logits[:, :max_components, :].squeeze(-1)

        # 1. Cross-entropy loss for existence prediction
        existence_loss = F.binary_cross_entropy_with_logits(existence_logits, mog_params["existence"])

        # 2. Gaussian NLL for mean and log-variance prediction for existing components
        nll_loss = 0
        for i in range(existence_logits.shape[0]):  # Iterate over batch
            for j in range(existence_logits.shape[1]):  # Iterate over components
                if mog_params["existence"][i, j] == 1:
                    dist = torch.distributions.Normal(pred_means[i, j], torch.exp(0.5 * pred_logvars[i, j]))
                    nll = -dist.log_prob(mog_params["means"][i, j]).sum()  # Sum over dimensions
                    nll_loss += nll

        # Normalize NLL loss by the number of existing components
        num_existing_components = mog_params["existence"].sum()
        if num_existing_components > 0:
            nll_loss = nll_loss / num_existing_components

        total_loss = existence_loss + nll_loss
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def prepare_data(self):
        # Generate synthetic data
        train_mog_params, train_samples = generate_gaussian_mixture(
            batch_size=10000, min_components=self.hparams.min_components, max_components=self.hparams.max_components,
            dim_output=self.hparams.dim_output, min_dist=self.hparams.min_dist,
            min_logvar=self.hparams.min_logvar, max_logvar=self.hparams.max_logvar,
            num_samples=self.hparams.num_samples
        )
        val_mog_params, val_samples = generate_gaussian_mixture(
            batch_size=500, min_components=self.hparams.min_components, max_components=self.hparams.max_components,
            dim_output=self.hparams.dim_output, min_dist=self.hparams.min_dist,
            min_logvar=self.hparams.min_logvar, max_logvar=self.hparams.max_logvar,
            num_samples=self.hparams.num_samples
        )
        test_mog_params, test_samples = generate_gaussian_mixture(
            batch_size=500, min_components=self.hparams.min_components, max_components=self.hparams.max_components,
            dim_output=self.hparams.dim_output, min_dist=self.hparams.min_dist,
            min_logvar=self.hparams.min_logvar, max_logvar=self.hparams.max_logvar,
            num_samples=self.hparams.num_samples
        )

        # Convert dictionaries of tensors to individual tensors
        train_mog_params = self._convert_to_tensors(train_mog_params)
        val_mog_params = self._convert_to_tensors(val_mog_params)
        test_mog_params = self._convert_to_tensors(test_mog_params)

        # Create TensorDatasets and DataLoaders
        self.train_dataset = TensorDataset(*train_mog_params, train_samples)
        self.val_dataset = TensorDataset(*val_mog_params, val_samples)
        self.test_dataset = TensorDataset(*test_mog_params, test_samples)

    def _convert_to_tensors(self, mog_params_dict):
        return (mog_params_dict['num_components'], mog_params_dict['means'], mog_params_dict['logvars'], mog_params_dict['existence'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

# Example Usage
if __name__ == "__main__":
    # Initialize wandb logger
    wandb_logger = WandbLogger(project="amortized-mog-fitting", log_model=True)

    # Add EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=10,         # Stop after 10 epochs with no improvement
        mode="min",          # Look for minimum validation loss
        verbose=True         # Print messages when early stopping happens
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        logger=wandb_logger,  # Use wandb_logger
        accelerator="cpu",
        callbacks=[early_stopping]  # Add the callback
    )

    model = MoGTrainer(
        dim_input=2, dim_output=2, dim_hidden=64, num_heads=4, num_blocks=2, max_components=5,
        min_components=1, min_dist=2.0, min_logvar=-2.0, max_logvar=2.0, num_samples=100
    )

    trainer.fit(model)
    trainer.test(model)

    wandb.finish()