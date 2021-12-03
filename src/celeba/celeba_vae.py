""" Contains code for the CelebA VAE model """

import torch
from torch import nn

from src.models import BaseVAE, UnFlatten


class CelebaVAE(BaseVAE):
    """ Convolutional VAE for encoding/decoding 64x64 CelebA images """

    def __init__(self, hparams):
        super().__init__(hparams)

        # Set up encoder and decoder
        self.encoder = nn.Sequential(
            # Many convolutions
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1  # out_size: 32x32x32
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1  # out_size: 64x16x16
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1  # out_size: 128x8x8
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1  # out_size: 256x4x4
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1  # out_size: 512x2x2
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            # Flatten and FC layers
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=2 * self.latent_dim),
        )

        self.decoder = nn.Sequential(
            # FC layers
            nn.Linear(in_features=self.latent_dim, out_features=2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            # Unflatten
            UnFlatten(512, 2),
            # Conv transpose layers
            nn.ConvTranspose2d(  # out_size: 256x4x4
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(  # out_size: 128x8x8
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(  # out_size: 64x16x16
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(  # out_size: 32x32x32
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(  # out_size: 32x64x64
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32, out_channels=3, kernel_size=3, padding=1  # out_size: 3x64x64
            ),
        )

    def encode_to_params(self, x):
        """ forward pass through encoder network """
        enc_output = self.encoder(x.float())
        mu, logstd = enc_output[:, : self.latent_dim], enc_output[:, self.latent_dim:]
        return mu, logstd

    def encode_to_latent_space(self, x):
        """ encode data from the original input space to latent space """
        mu, logstd = self.encode_to_params(x.float())
        encoder_distribution = torch.distributions.Normal(
            loc=mu, scale=torch.exp(logstd)
        )
        z_sample = encoder_distribution.rsample()
        return z_sample

    def decoder_loss(self, z, x_orig):
        """ forward pass through decoder network with loss computation """
        logits = self.decoder(z)
        x_recon = torch.sigmoid(logits)
        mse_loss = nn.MSELoss(reduction='sum')
        return mse_loss(x_recon, x_orig) / (z.shape[0])

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        """ maps deterministically from latent space to input space """
        logits = self.decoder(z)
        return torch.sigmoid(logits)

