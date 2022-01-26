""" Contains code for the CelebA VQ-VAE model with m = 8 """

import torch
from torch import nn
from torch.nn import functional as F
import argparse
import pytorch_lightning as pl

#import distributed as dist_fn


class Quantize(nn.Module):
    """
    Reference:
    https://github.com/rosinality/vq-vae-2-pytorch
    """
    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 decay: float = 0.99,
                 eps=1e-5):
        super(Quantize, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(self.D, self.K)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.D)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.K).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            #torch.distributed.all_reduce(embed_onehot_sum)
            #torch.distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


    # def encode_latents_to_discrete_latent_space(self, latents):
    #     latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
    #     flat_latents = latents.view(-1, self.D)  # [BHW x D]
    #
    #     # Compute L2 distance between latents and embedding weights
    #     dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
    #            torch.sum(self.embedding.weight ** 2, dim=1) - \
    #            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
    #
    #     # Get the encoding that has the min distance
    #     z = torch.argmin(dist, dim=1).unsqueeze(1).view(latents.shape[0], -1)  # [B, HW]
    #
    #     return latents, z

class ResBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 channel: int):
        super().__init__()
        self.resblock = nn.Sequential(torch.nn.ReLU(),
                                      nn.Conv2d(in_channel, channel,
                                                kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(channel, in_channel,
                                                kernel_size=1))

    def forward(self, input):
        return input + self.resblock(input)


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ] # enc_b: 128x16x16

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ] # enc_t: 128x8x8

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
            # enc_b: 128x16x16

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]  # 128x8x8 or 128x16x16

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))  # 128x8x8 or 128x16x16

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),  # 64x32x32
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1  # 3x64x64
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )  # 64x16x16

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class CelebaVQVAE2(pl.LightningModule):
    def __init__(
        self,
        hparams
    ):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.embed_dim = hparams.embedding_dim
        self.n_embed = hparams.num_embeddings
        self.beta = hparams.beta

        self.logging_prefix = None
        self.log_progress_bar = False

        self.enc_b = Encoder(3, 128, 2, 64, stride=4)
        self.enc_t = Encoder(128, 128, 2, 64, stride=2)
        self.quantize_conv_t = nn.Conv2d(128, self.embed_dim, 1)
        self.quantize_t = Quantize(self.embed_dim, self.n_embed)
        self.dec_t = Decoder(
            self.embed_dim, self.embed_dim, 128, 2, 64, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(self.embed_dim + 128, self.embed_dim, 1)
        self.quantize_b = Quantize(self.embed_dim, self.n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            self.embed_dim, self.embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            self.embed_dim + self.embed_dim, 3, 128, 2, 64, stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):  # input: 3x64x64
        enc_b = self.enc_b(input)  # 128x16x16
        enc_t = self.enc_t(enc_b)  # 128x8x8

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)  # 64x8x8
        quant_t, diff_t, id_t = self.quantize_t(quant_t)  # 64x8x8
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)  # 64x16x16
        enc_b = torch.cat([dec_t, enc_b], 1)  # 192x16x16

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)  # 64x16x16
        quant_b, diff_b, id_b = self.quantize_b(quant_b)  # 64x16x16
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)  # 64x16x16
        quant = torch.cat([upsample_t, quant_b], 1)  # 128x16x16
        dec = self.dec(quant)  # 3x64x64

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        vqvae2_group = parser.add_argument_group("VQ-VAE2")
        vqvae2_group.add_argument("--lr", type=float, default=3e-4)
        vqvae2_group.add_argument("--num_embeddings", type=int, default=512)
        vqvae2_group.add_argument("--embedding_dim", type=int, default=64)
        vqvae2_group.add_argument("--beta", type=float, default=0.25)
        return parser

    def loss_function(self, input, output, latent_loss):
        """
        :param args:
        :param kwargs:
        :return:
        """

        recon_loss = F.mse_loss(output, input)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.beta * latent_loss

        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                recon_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"vq/{self.logging_prefix}", latent_loss, prog_bar=self.log_progress_bar
            )
            self.log(f"loss/{self.logging_prefix}", loss)
        return {'loss': loss,
                'Reconstruction_Loss': recon_loss,
                'VQ_Loss': latent_loss}

    def training_step(self, batch, batch_idx):
        self.logging_prefix = "train"
        self.log_progress_bar = True
        out, latent_loss = self(batch[0])
        train_loss = self.loss_function(batch[0], out, latent_loss)
        self.logging_prefix = None
        self.log_progress_bar = False

        return train_loss

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        out, latent_loss = self(batch[0])
        val_loss = self.loss_function(batch[0], out, latent_loss)
        self.logging_prefix = None
        self.log_progress_bar = False

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # def reconstruct(self, x):
    #     """
    #     Given an input image x, returns the reconstructed image
    #     :param x: (Tensor) [B x C x H x W]
    #     :return: (Tensor) [B x C x H x W]
    #     """
    #
    #     return self(x)[0]
    #
    # def encode_to_latent_space(self, x):
    #     """ encode data from the original input space to latent space """
    #     latents = self.encode(x)[0]
    #
    #     latents, z = self.vq_layer.encode_latents_to_discrete_latent_space(latents)
    #
    #     return z
    #
    # def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
    #
    #     z_rearr = z.view(-1, 1)
    #     # Convert to one-hot encodings
    #     encoding_one_hot = torch.zeros(z_rearr.shape[0], self.num_embeddings, device=self.device)
    #     encoding_one_hot.scatter_(1, z_rearr, 1)  # [BHW x K]
    #
    #     # Quantize the latents
    #     quantized_latents = torch.matmul(encoding_one_hot, self.vq_layer.embedding.weight) # [BHW, D]
    #     quantized_latents = quantized_latents.view((z.shape[0], 8, 8, self.embedding_dim))  # [B x H x W x D]
    #     x = self.decode(quantized_latents.permute(0, 3, 1, 2).contiguous())
    #
    #     return x
