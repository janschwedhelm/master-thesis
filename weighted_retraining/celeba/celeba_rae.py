""" Contains code for the binary MNIST VAE model """

import itertools
import numpy as np
import torch
from torch import nn, distributions
from torchvision.utils import make_grid

# My imports
from weighted_retraining.models import UnFlatten
from weighted_retraining.rae import RAE


class CelebaRAE(RAE):
    """ Convolutional VAE for encoding/decoding 128x128 CelebA images """

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
            nn.Linear(in_features=2048, out_features=self.latent_dim),
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

    def decoder_loss(self, z, x_orig):  # forward pass through decoder network with loss computation
        """ return negative Bernoulli log prob """
        logits = self.decoder(z)
        x_recon = torch.sigmoid(logits)
        mse_loss = nn.MSELoss(reduction='sum')
        #print(f"x reconstruction: {x_recon}")
        #print(f"original input: {x_orig}")
        #print(f"reconstruction shape: {x_recon.shape}")
        #print(f"z shape: {z.shape}")
        #print(f"MSE loss: {mse_loss(x_recon, x_orig)}")
        #print(x_orig.shape[1] * x_orig.shape[2] * x_orig.shape[3])
        return mse_loss(x_recon, x_orig) / (z.shape[0])

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        return torch.sigmoid(logits)
    
    #def sample(self,
    #           num_samples:int):
    #    """
    #    Samples from the latent space and return the corresponding
    #    image space map.
    #    :param num_samples: (Int) Number of samples
    #    :param current_device: (Int) Device to run the model
    #    :return: (Tensor)
    #    """
    #    z = torch.randn(num_samples,
    #                    self.latent_dim)
    #
    #    #z = z.to(current_device)

    #    samples = self.decode_deterministic(z)
    #    return samples
