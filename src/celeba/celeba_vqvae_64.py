""" Contains code for the CelebA VQ-VAE model with m = 8 """

import torch
from torch import nn
from torch.nn import functional as F
import argparse
import pytorch_lightning as pl


class VectorQuantizer(nn.Module):
    """
    Reference:
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents, encoding_inds = self.encode_latents_to_discrete_latent_space(latents)
        encoding_inds = encoding_inds.view(-1,1) # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.shape[0], self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents.shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

    def encode_latents_to_discrete_latent_space(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        z = torch.argmin(dist, dim=1).unsqueeze(1).view(latents.shape[0], -1)  # [B, HW]

        return latents, z


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input):
        return input + self.resblock(input)


class CelebaVQVAE(pl.LightningModule):

    def __init__(self,
                 hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.embedding_dim = hparams.embedding_dim
        self.num_embeddings = hparams.num_embeddings
        self.beta = hparams.beta

        self.logging_prefix = None
        self.log_progress_bar = False

        modules = []
        hidden_dims = [64, 128, 256]
        in_channels = 3

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(2):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, self.embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(self.num_embeddings,
                                        self.embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(self.embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU(),
        ))

        for _ in range(2):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                ))

        self.decoder = nn.Sequential(*modules)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        vqvae_group = parser.add_argument_group("VQ-VAE")
        vqvae_group.add_argument("--lr", type=float, default=1e-3)
        vqvae_group.add_argument("--beta", type=float, default=0.25)
        vqvae_group.add_argument("--num_embeddings", type=int, default=256)
        vqvae_group.add_argument("--embedding_dim", type=int, default=64)
        return parser

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        logits = self.decoder(z)
        return torch.sigmoid(logits)

    def forward(self, x):
        encoding = self.encode(x)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), x, vq_loss]

    def loss_function(self,
                      results):
        recons = results[0]
        input = results[1]
        vq_loss = results[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                recons_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"vq/{self.logging_prefix}", vq_loss, prog_bar=self.log_progress_bar
            )
            self.log(f"loss/{self.logging_prefix}", loss)
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def training_step(self, batch, batch_idx):
        self.logging_prefix = "train"
        self.log_progress_bar = True
        results = self(batch[0])
        train_loss = self.loss_function(results)
        self.logging_prefix = None
        self.log_progress_bar = False

        return train_loss

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        results = self(batch[0])
        val_loss = self.loss_function(results)
        self.logging_prefix = None
        self.log_progress_bar = False

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def reconstruct(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self(x)[0]

    def encode_to_latent_space(self, x):
        """ encode data from the original input space to latent space """
        latents = self.encode(x)[0]

        latents, z = self.vq_layer.encode_latents_to_discrete_latent_space(latents)

        return z

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:

        z_rearr = z.view(-1, 1)
        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(z_rearr.shape[0], self.num_embeddings, device=self.device)
        encoding_one_hot.scatter_(1, z_rearr, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.vq_layer.embedding.weight) # [BHW, D]
        quantized_latents = quantized_latents.view((z.shape[0], 8, 8, self.embedding_dim))  # [B x H x W x D]
        x = self.decode(quantized_latents.permute(0, 3, 1, 2).contiguous())

        return x
