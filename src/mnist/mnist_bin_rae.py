""" Contains code for the binary MNIST RAE model """

import torch
from torch import nn, distributions

from src.rae import RAE
from src.models import UnFlatten


class MnistBinRAE(RAE):
    """ Convolutional RAE for encoding/decoding 28x28 binary images """

    def __init__(self, hparams):
        super().__init__(hparams)

        #Set up encoder and decoder
        self.encoder = nn.Sequential(
            # Many convolutions
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1  # out_size: 128x16x16
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1  # out_size: 256x8x8
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1  # out_size: 512x4x4
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1  # out_size: 1024x2x2
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # Flatten and FC layers
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            # FC layers
            nn.Linear(in_features=self.latent_dim, out_features=65536),
            nn.BatchNorm1d(65536),
            nn.ReLU(),
            # Unflatten
            UnFlatten(1024, 8),
            # Conv transpose layers
            nn.ConvTranspose2d(  # out_size: 512x16x16
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                padding=1,
                stride=2,
                output_padding=0,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(  # out_size: 256x32x32
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                padding=1,
                stride=2,
                output_padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(  # out_size: 1x32x32
                in_channels=256,
                out_channels=1,
                kernel_size=4,
                padding=3,
                stride=1,
                output_padding=0,
                dilation=2
            ),
        )

        self.encoder = nn.Sequential(
            # Many convolutions
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1  # out_size: 32x28x28
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1  # out_size: 64x14x14
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1  # out_size: 64x7x7
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1  # out_size: 64x7x7
            ),
            nn.ReLU(),
            # Flatten and FC layers
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.latent_dim),
        )

        self.decoder = nn.Sequential(
            # FC layers
            nn.Linear(in_features=self.latent_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=3136),
            nn.ReLU(),
            # Unflatten
            UnFlatten(64, 7),
            # Conv transpose layers
            nn.ConvTranspose2d(  # out_size: 64x7x7
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(  # out_size: 64x14x14
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(  # out_size: 32x28x28
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(  # out_size: 1x28x28
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
        )

    def decoder_loss(self, z, x_orig):  # forward pass through decoder network with loss computation
        """ return negative Bernoulli log prob """
        logits = self.decoder(z)
        dist = distributions.Bernoulli(logits=logits)
        return -dist.log_prob(x_orig).sum() / z.shape[0]

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        return torch.sigmoid(logits)
