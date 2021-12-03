""" code for MNIST classifier """

import argparse
import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics


class MNISTClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # call this to save hparams to checkpoint
        self.save_hyperparameters()

        # Set up CNN
        self.cnn = nn.Sequential(
            # Many convolutions
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1  # out_size: 32x28x28
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1  # out_size: 32x28x28
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2  # out_size: 32x14x14
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1  # out_size: 64x14x14
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1  # out_size: 64x14x14
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2  # out_size: 64x7x7
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            # Flatten and FC layers
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(in_features=128, out_features=10),
        )

        # Define activation function on output layer
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Define loss function
        self.loss_function = nn.NLLLoss()

        # Define further metrics to log
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        model_group = parser.add_argument_group("model")
        model_group.add_argument("--lr", type=float, default=1e-3)
        return parser

    def forward(self, x):
        x = self.cnn(x)
        #x = self.logsoftmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        log_probs = self.logsoftmax(logits)
        self.log('acc/train', self.train_acc(log_probs, y), on_step=True, on_epoch=True, prog_bar=True)
        loss = self.loss_function(log_probs, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        log_probs = self.logsoftmax(logits)
        loss = self.loss_function(log_probs, y)
        self.log(
            f"loss/val",
            loss,
            prog_bar=True,
        )
        self.log('acc/val', self.val_acc(log_probs, y), on_step=True, on_epoch=True, prog_bar=True)
        # self.accuracy.update(logits.argmax(dim=-1), y)
        # if self.trainer.accumulate_grad_batches % self.global_step == 0:
        #     accumulated_val = self.accuracy.compute()
        #     self.log('acc_accumulate', accumulated_val)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

