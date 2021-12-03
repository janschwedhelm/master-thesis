""" Code for base RAE model """

import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl


class RAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # call this to save hparams to checkpoint
        self.save_hyperparameters()
        self.latent_dim = hparams.latent_dim
        self.latent_emb_weight = hparams.latent_emb_weight
        self.reg_weight = hparams.reg_weight
        self.reg_type = hparams.reg_type

        self.logging_prefix = None
        self.log_progress_bar = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        rae_group = parser.add_argument_group("RAE")
        rae_group.add_argument("--latent_dim", type=int, required=True)
        rae_group.add_argument("--lr_start", type=float, default=1e-3)
        rae_group.add_argument("--latent_emb_weight", type=float, default=0.1)
        rae_group.add_argument("--reg_type", type=str, default="l2")
        rae_group.add_argument("--reg_weight", type=float, default=1e-4)

        return parser

    def forward(self, x):
        """ calculate the RAE loss function """
        # Compute reconstruction loss
        z = self.encode_to_latent_space(x)
        reconstruction_loss = self.decoder_loss(z, x)

        # Compute latent embedding loss term
        latent_emb_loss = z.pow(2).sum() / z.shape[0]

        # Compute regularization loss term
        if self.reg_type == "l2":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            regularization_loss = torch.tensor(0.0).to(device)
            for param in self.decoder.parameters():
                regularization_loss += torch.norm(param.to(device)).pow(2)
        else:
            raise NotImplementedError(self.reg_type)

        # Compute total loss
        loss = reconstruction_loss + self.latent_emb_weight * latent_emb_loss + self.reg_weight * regularization_loss

        # Logging
        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                reconstruction_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"reg/{self.logging_prefix}", regularization_loss, prog_bar=self.log_progress_bar
            )
            self.log(
                f"emb/{self.logging_prefix}", latent_emb_loss, prog_bar=self.log_progress_bar
            )
            self.log(f"loss/{self.logging_prefix}", loss, prog_bar=self.log_progress_bar)
        return loss

    def encode_to_latent_space(self, x):
        """ encode data from the original input space to latent space """
        z = self.encoder(x.float())
        return z

    # Method to overwrite (differs between specific RAE implementations)
    def decoder_loss(self, z: torch.Tensor, x_orig) -> torch.Tensor:
        """ Get the loss of the decoder given a batch of z values to decode """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self.logging_prefix = "train"
        self.log_progress_bar = True
        loss = self(batch[0])
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        loss = self(batch[0])
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr_start)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=5),
                "monitor": "loss/val",
            },
        }
