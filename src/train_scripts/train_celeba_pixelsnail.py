""" Trains a PixelSNAIL with VQ-VAE2 for the CelebA task """

import argparse
import pytorch_lightning as pl

from src.celeba.celeba_pixelsnail import PixelSNAIL
from src.dataloader_celeba_weighting import CelebaWeightedTensorDataset, SimpleFilenameToTensorDataset
from src import utils


def main(args):
    # Load model
    if args.hier == 'top':
        model = PixelSNAIL(args, attention=True)
    else:
        model = PixelSNAIL(args, attention=False)
    if hparams.load_from_checkpoint is not None:
        model = PixelSNAIL.load_from_checkpoint(hparams.load_from_checkpoint)
        utils.update_hparams(args, model)

    pl.seed_everything(args.seed)

    # Create data
    datamodule = CelebaWeightedTensorDataset(args, utils.DataWeighter(args))

    # Main trainer
    trainer = pl.Trainer(
        gpus=1 if args.gpu else 0,
        default_root_dir=args.root_dir,
        max_epochs=args.max_epochs,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            period=10, monitor="loss/val", save_top_k=-1,  # save models after every 10 epochs of training
            save_last=True
        ),
        terminate_on_nan=True,
        min_epochs=1,
        num_sanity_val_steps=5,
    )

    # Fit
    trainer.fit(model, datamodule=datamodule)
    print("Training finished; end of script")


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()
    # add model specific args
    parser = PixelSNAIL.add_model_specific_args(parser)
    # add data specific args
    parser = CelebaWeightedTensorDataset.add_model_specific_args(parser)
    # add weighting specific args
    parser = utils.DataWeighter.add_weight_args(parser)
    #add trainer specific args
    utils.add_default_trainer_args(parser, default_root="logs/train/CelebaPixelSNAIL")

    # Parse arguments
    hparams = parser.parse_args()

    main(hparams)
