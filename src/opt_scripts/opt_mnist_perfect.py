""" Run weighted retraining for MNIST task using 'perfect' optimizer strategy """

import sys
import logging
import itertools
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import ruamel.yaml
from sklearn.mixture import GaussianMixture

from src.dataloader_weighting import WeightedNumpyDataset
from src.mnist.mnist_bin_vae import MnistBinVAE
from src.mnist.mnist_bin_rae import MnistBinRAE
from src import utils
from src.opt_scripts import base as wr_base

from src.utils import sparse_subset


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu):
    """
    helper function to retrain model
    """
    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)
    train_pbar = utils.SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val",)

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")

    # Create trainer
    trainer = pl.Trainer(
        gpus=1 if gpu else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=checkpointer,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar],
    )

    # Fit model
    trainer.fit(model, datamodule)


def _batch_decode_z_and_props(model, z, args, get_props=True, filter_unique=True):
    """
    helper function to decode some latent vectors and calculate their properties
    """
    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1000
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = model.decode_deterministic(z[j : j + batch_size])
            img = img.cpu().numpy()
            z_decode.append(img)
            del img

    # Concatentate all points and convert to numpy
    z_decode = np.concatenate(z_decode, axis=0)
    z_decode = np.around(z_decode)  # convert to int
    z_decode = z_decode[:, 0, ...]  # Correct slicing
    if filter_unique:
        z_decode, uniq_indices = np.unique(
            z_decode, axis=0, return_index=True
        )  # Unique elements only
        z = z.cpu().numpy()[uniq_indices]

    # Calculate objective function values and choose which points to keep
    if get_props:
        if args.property_key == "thickness":
            z_prop = np.mean(z_decode, axis=(-1, -2))
        else:
            raise ValueError(args.property_key)

    if filter_unique:
        if get_props:
            return z_decode, z_prop, z
        else:
            return z_decode, z
    else:
        if get_props:
            return z_decode, z_prop
        else:
            return z_decode


def latent_optimization(args, model, datamodule, num_queries_to_do):
    """
    perform latent space optimization using 'perfect' optimization strategy
    """
    unit_line = np.linspace(-args.opt_bounds, args.opt_bounds, args.opt_grid_len)  # default length: 50
    latent_grid = list(itertools.product(unit_line, repeat=model.latent_dim))  # default length: 2500
    latent_grid = np.array(latent_grid, dtype=np.float32)

    if args.opt_constraint_threshold is None:
        z_latent_opt = torch.as_tensor(latent_grid, device=model.device)
    else:
        # Determine valid latent points for sampling
        # using a full GMM where each encoded training sample serves as a component
        if args.opt_constraint_strategy == "gmm_full":
            mu, logstd = model.encode_to_params(torch.tensor(datamodule.data_train).unsqueeze(1))
            mu = mu.detach().numpy()
            logstd = logstd.detach().numpy()
            variances = np.exp(2 * logstd)
            logdens_z_grid = utils.log_gmm_density(latent_grid, mu, variances)
        # using a fitted GMM using a random sample for each encoded distribution from the training data
        elif args.opt_constraint_strategy == "gmm_fit":
            if not args.opt_constraint_threshold:
                raise Exception("Please specify a log-density threshold under the GMM model if "
                                "'gmm_fit' is used as optimization constraint strategy.")
            if not args.n_gmm_components:
                raise Exception("Please specify number of components to use for the GMM model if "
                                "'gmm_fit' is used as optimization constraint strategy.")
            z_train = model.encode_to_latent_space(torch.tensor(datamodule.data_train).unsqueeze(1))
            gmm = GaussianMixture(n_components=args.n_gmm_components, random_state=0).fit(z_train.detach().numpy())
            logdens_z_grid = gmm.score_samples(latent_grid)
        else:
            raise NotImplementedError(args.opt_constraint_strategy)

        valid_z = np.array([z for i, z in enumerate(latent_grid) if logdens_z_grid[i] > args.opt_constraint_threshold],
                           dtype=np.float32)
        z_latent_opt = torch.as_tensor(valid_z, device=model.device)

    # Decodes points from latent grid and evaluates these decoded points (unique)
    z_decode, z_prop, z = _batch_decode_z_and_props(model, z_latent_opt, args)

    z_prop_argsort = np.argsort(-1 * z_prop)  # assuming maximization of property

    # Choose new points
    z_decode_sorted = z_decode[z_prop_argsort]
    z_prop_sorted = z_prop[z_prop_argsort]
    z_sorted = z[z_prop_argsort]
    z_decode_sorted, index = sparse_subset(z_decode_sorted, 0.1)
    new_points = z_decode_sorted[:num_queries_to_do]
    y_new = z_prop_sorted[index][:num_queries_to_do]
    z_query = z_sorted[index][:num_queries_to_do]
    return new_points, y_new, z_query


def main_loop(args):

    # Seeding
    pl.seed_everything(args.seed)

    # Load data
    datamodule = WeightedNumpyDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit") # assignment into train/validation/split is made and weights are set

    # Load pre-trained model
    if args.pretrained_model_type == "vae":
        model = MnistBinVAE.load_from_checkpoint(args.pretrained_model_file)
        model.beta = model.hparams.beta_final  # Override any beta annealing
    elif args.pretrained_model_type == "rae":
        model = MnistBinRAE.load_from_checkpoint(args.pretrained_model_file)
    else:
        raise NotImplementedError(args.pretrained_model_type)

    # Set up results tracking
    results = dict(
        opt_points=[],  # saves (default: 5) optimal points in the original input space for each retraining iteration
        opt_latent_points=[],  # saves corresponding points in the latent space
        opt_point_properties=[],  # saves corresponding function evaluations
        opt_model_version=[],  # saves corresponding retraining iteration
        params=str(sys.argv),
        latent_space_snapshots=[],  # saves decoded latent space points uniformly enumerated on [-4,4]^2 for each retraining iteration
        latent_space_snapshot_version=[],  # saves corresponding retraining iteration
    )

    # Set up latent space snapshot!
    # length: 289
    results["latent_space_grid"] = np.array(
        list(itertools.product(np.arange(-4, 4.01, 0.5), repeat=model.latent_dim)),
        dtype=np.float32,
    )

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))  # default: 500/5 = 100
    postfix = dict(
        retrain_left=num_retrain, best=-float("inf"), n_train=len(datamodule.data_train)
    )

    # Main loop
    with tqdm(
        total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        # Make result directory
        result_dir = Path(args.result_root).resolve()
        result_dir.mkdir(parents=True)
        data_dir = result_dir / "data"
        data_dir.mkdir()

        # Save retraining hyperparameters
        with open(result_dir / 'retraining_hparams.yaml', 'w') as f:
            ruamel.yaml.dump(args.__dict__, f, default_flow_style=False)

        for ret_idx in range(num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain
            samples_so_far = args.retraining_frequency * ret_idx

            # Optionally do retraining
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                # default: initial fine-tuning of pre-trained model for 1 epoch
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = result_dir / "retraining"
                version = f"retrain_{samples_so_far}"
                # default: run through 10% of the weighted training data in retraining epoch
                retrain_model(
                    model, datamodule, retrain_dir, version, num_epochs, args.gpu
                )

            # Take latent snapshot
            # decodes latent points uniformly enumerated on [-4,4]^2
            latent_snapshot = _batch_decode_z_and_props(
                model,
                torch.as_tensor(results["latent_space_grid"], device=model.device),
                args,
                filter_unique=False,
            )[0]
            results["latent_space_snapshots"].append(latent_snapshot)
            results["latent_space_snapshot_version"].append(ret_idx)

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)
            pbar.set_description("querying")

            # Do querying!
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )

            # default: find 5 latent points from latent space grid that lead to highest objective function values
            x_new, y_new, z_query = latent_optimization(
                args, model, datamodule, num_queries_to_do
            )

            # Append new points to dataset and adapt weighting
            datamodule.append_train_data(x_new, y_new)

            # Save a new dataset
            new_data_file = (
                data_dir / f"train_data_iter{samples_so_far + num_queries_to_do}.npz"
            )
            np.savez_compressed(
                str(new_data_file),
                data=datamodule.data_train,
                **{args.property_key: datamodule.prop_train},
            )

            # Save results
            results["opt_latent_points"] += [z for z in z_query]
            results["opt_points"] += [x for x in x_new]
            results["opt_point_properties"] += [y for y in y_new]
            results["opt_model_version"] += [ret_idx] * len(x_new)
            np.savez_compressed(str(result_dir / "results.npz"), **results)

            # Final update of progress bar
            postfix["best"] = max(postfix["best"], float(y_new.max()))
            postfix["n_train"] = len(datamodule.data_train)
            pbar.set_postfix(postfix)
            pbar.update(n=num_queries_to_do)
    print("Weighted retraining finished; end of script")


if __name__ == "__main__":

    # arguments and argument checking
    parser = argparse.ArgumentParser()
    parser = WeightedNumpyDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    parser = wr_base.add_common_args(parser)

    # Optimal model arguments
    opt_group = parser.add_argument_group(title="opt-model")
    opt_group.add_argument("--opt_bounds", type=float, default=4)
    opt_group.add_argument("--opt_grid_len", type=float, default=50)
    opt_group.add_argument("--opt_constraint_threshold", type=float, default=None)
    opt_group.add_argument("--opt_constraint_strategy", type=str, default="gmm_full")
    opt_group.add_argument("--n_gmm_components", type=int, default=None)

    args = parser.parse_args()

    main_loop(args)
