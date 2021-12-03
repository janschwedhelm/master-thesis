""" Creates a dataloader for training a CNN classifier on MNIST digits"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

NUM_WORKERS = 2


class ImageClassifierDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.dataset_path = hparams.dataset_path
        self.val_frac = hparams.val_frac
        self.batch_size = hparams.batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument(
            "--dataset_path", type=str, required=True, help="path to npz file"
        )

        data_group.add_argument("--batch_size", type=int, default=32)
        data_group.add_argument(
            "--val_frac",
            type=float,
            default=0.05,
            help="Fraction of val data",
        )

        return parent_parser

    @staticmethod
    def _get_tensor_dataset(X, y):
        X = torch.as_tensor(X, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.long)
        X = torch.unsqueeze(X, 1) # insert dimension for number of channels in image
        return TensorDataset(X, y)

    def setup(self, stage):
        #make assignments here: train/val split
        with np.load(self.dataset_path) as npz:
            all_data = npz["data"]
            all_targets = npz["targets"]
        assert all_targets.shape[0] == all_data.shape[0]

        N_val = int(all_data.shape[0] * self.val_frac)
        self.data_val = all_data[:N_val]
        self.target_val = all_targets[:N_val]
        self.data_train = all_data[N_val:]
        self.target_train = all_targets[N_val:]

        # Transform into tensor datasets
        self.train_dataset = ImageClassifierDataModule._get_tensor_dataset(self.data_train, self.target_train)
        self.val_dataset = ImageClassifierDataModule._get_tensor_dataset(self.data_val, self.target_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS
        )
