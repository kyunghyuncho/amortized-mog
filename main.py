from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import wandb

from src.amortized_mixture_of_gaussians import AmortizedMixtureOfGaussians

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
        max_epochs=2,
    )

    model = AmortizedMixtureOfGaussians(
        dim_output=2,
        dim_hidden=64,
        num_heads=4,
        num_blocks=6,
        components=3,
        dropout=0.1,
        maximum_components=5,
        minimum_components=1,
        minimum_distance=2.0,
        minimum_log_variance=-2.0,
        maximum_log_variance=2.0,
        num_samples=100,
        check_test_loss_every_n_epoch=1,
        lr=1e-3,
    )

    trainer.fit(model)
    trainer.test(model)

    wandb.finish()
