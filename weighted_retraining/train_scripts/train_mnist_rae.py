""" Trains a RAE for the MNIST task """

import argparse
import pytorch_lightning as pl

from weighted_retraining.mnist.mnist_bin_rae import MnistBinRAE
from weighted_retraining.dataloader_weighting import WeightedNumpyDataset
from weighted_retraining import utils


def main(args):
    pl.seed_everything(args.seed)
    
    # Load model
    model = MnistBinRAE(args)
    if hparams.load_from_checkpoint is not None:
        model = MnistBinRAE.load_from_checkpoint(hparams.load_from_checkpoint)
        utils.update_hparams(args, model)

    # Create data
    datamodule = WeightedNumpyDataset(args, utils.DataWeighter(args))

    # Main trainer
    trainer = pl.Trainer(
        gpus=1 if args.gpu else 0,
        default_root_dir=args.root_dir,
        max_epochs=args.max_epochs,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            period=10, monitor="loss/val", save_top_k=-1,  # save models after every 10 epochs of training
            save_last=True
        ),
        terminate_on_nan=True
    )

    # Fit
    trainer.fit(model, datamodule=datamodule)
    print("Training finished; end of script")


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()
    # add model specific args
    parser = MnistBinRAE.add_model_specific_args(parser)
    # add data specific args
    parser = WeightedNumpyDataset.add_model_specific_args(parser)
    # add weighting specific args
    parser = utils.DataWeighter.add_weight_args(parser)
    #add trainer specific args
    utils.add_default_trainer_args(parser, default_root="logs/train/mnist_2")

    # Parse arguments
    hparams = parser.parse_args()

    main(hparams)
