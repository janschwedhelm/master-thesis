""" Borrowed from https://github.com/ritheshkumar95/pytorch-vqvae """

import torch
import torch.nn as nn
import numpy as np
import argparse
import pytorch_lightning as pl

from src.celeba.celeba_vqvae_64 import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1) # splits feature maps in two chunks for tanh, and sigmoid, respectively
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    # see https://sergeiturukin.com/2017/02/24/gated-pixelcnn.html for technical details
    #def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
    def __init__(self, mask_type, dim, kernel, residual=True):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        kernel_shp = (kernel // 2 + 1, kernel)  # 3x3 -> 2x3, 7x7 -> 4x7
        padding_shp = (kernel // 2, kernel // 2)  # 3x3 -> (1,1), 7x7 -> (3,3)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2, # *2 for Gated Activation
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1) # 3x3 -> 1x2, 7x7 -> 1x4
        padding_shp = (0, kernel // 2) # 3x3 -> (0,1), 7x7 -> (0,3)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    #def forward(self, x_v, x_h, h):
    def forward(self, x_v, x_h):
        if self.mask_type == 'A':
            self.make_causal()

        #h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]  # crop to get appropriate size
        #out_v = self.gate(h_vert + h[:, :, None, None])
        out_v = self.gate(h_vert)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]  # crop to get appropriate size
        v2h = self.vert_to_horiz(h_vert)

        #out = self.gate(v2h + h_horiz + h[:, :, None, None])
        out = self.gate(v2h + h_horiz)
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(pl.LightningModule):
    #def __init__(self, hparams, input_dim=256, dim=64, n_layers=15, n_classes=10):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.dim = hparams.dim
        self.input_dim = hparams.input_dim
        self.n_layers = hparams.n_layers
        self.vqvae_path = hparams.vqvae_path

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(self.input_dim, self.dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(self.n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                #GatedMaskedConv2d(mask_type, self.dim, kernel, residual, n_classes)
                GatedMaskedConv2d(mask_type, self.dim, kernel, residual)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, self.input_dim, 1)
        )

        self.apply(weights_init)

        # load corresponding VQ-VAE model and freeze parameters
        self.vq_vae = CelebaVQVAE.load_from_checkpoint(self.vqvae_path)
        for p in self.vq_vae.parameters():
            p.requires_grad = False
        self.vq_vae.eval()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        pixelcnn_group = parser.add_argument_group("PixelCNN")
        pixelcnn_group.add_argument("--vqvae_path", type=str, required=True)
        pixelcnn_group.add_argument("--lr", type=float, default=3e-4)
        pixelcnn_group.add_argument("--input_dim", type=int, default=256)
        pixelcnn_group.add_argument("--dim", type=int, default=64)
        pixelcnn_group.add_argument("--n_layers", type=int, default=15)
        return parser

    #def forward(self, x, label):
    def forward(self, x):
        # requires input shape (B, H, W)
        shp = x.size() + (-1, ) # (B, H, W, -1)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C) -> C = dim
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            #x_v, x_h = layer(x_v, x_h, label)
            x_v, x_h = layer(x_v, x_h)

        return self.output_conv(x_h)

    def loss_function(self,
                      latents):
        logits = self(latents)
        loss = logits.permute(0, 2, 3, 1).contiguous()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.input_dim), latents.view(-1))
        if self.logging_prefix is not None:
            self.log(f"loss/{self.logging_prefix}", loss)
        return loss

    def training_step(self, batch, batch_idx):
        self.logging_prefix = "train"
        self.log_progress_bar = True
        #results = self(batch[0])
        discrete_latents = self.vq_vae(batch[0])
        discrete_latents = discrete_latents.view(-1, 8, 8)
        train_loss = self.loss_function(discrete_latents)
        self.logging_prefix = None
        self.log_progress_bar = False

        return train_loss

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        #results = self(batch[0])
        discrete_latents = self.vq_vae(batch[0])
        discrete_latents = discrete_latents.view(-1, 8, 8)
        val_loss = self.loss_function(discrete_latents)
        self.logging_prefix = None
        self.log_progress_bar = False

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)

    #def generate(self, label, shape=(8, 8), batch_size=64):
    def generate(self, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                #logits = self.forward(x, label)
                logits = self.forward(x)
                probs = torch.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )  # Set pixel at position i, j
        return x
