""" Trains an SN-GAN for the CelebA task """

import argparse
import pytorch_lightning as pl

import torch
import torch.optim as optim
from src.torch_mimicry.training.trainer import Trainer
from src.torch_mimicry.nets.sngan.sngan_64 import SNGANGenerator64, SNGANDiscriminator64
from src.dataloader_celeba_weighting import CelebaWeightedTensorDataset, SimpleFilenameToTensorDataset
from src import utils


def main(args):

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Load model
    netG = SNGANGenerator64(nz=args.latent_dim).to(device)
    netD = SNGANDiscriminator64().to(device)
    optD = optim.Adam(netD.parameters(), args.lr, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), args.lr, betas=(0.0, 0.9))

    pl.seed_everything(args.seed)

    # Create data
    datamodule = CelebaWeightedTensorDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    # Main trainer
    trainer = Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=5,
        num_steps=80000,
        lr_decay='linear',
        dataloader=dataloader,
        log_dir=args.root_dir,
        netD_ckpt_file=args.checkpoint_discriminator,
        netG_ckpt_file=args.checkpoint_generator,
        device=device)
    trainer.train()

    print("Training finished; end of script")


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()
    # add model specific args
    parser = SNGANGenerator64.add_model_specific_args(parser)
    # add data specific args
    parser = CelebaWeightedTensorDataset.add_model_specific_args(parser)
    # add weighting specific args
    parser = utils.DataWeighter.add_weight_args(parser)
    #add trainer specific args
    utils.add_default_trainer_args(parser, default_root="logs/train/celeba/sn-gan")

    # Parse arguments
    hparams = parser.parse_args()

    main(hparams)
