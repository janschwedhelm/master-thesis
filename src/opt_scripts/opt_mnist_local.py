""" Run weighted retraining for MNIST task using traditional local optimization strategies """

import sys
import logging
import subprocess
import time
import itertools
from tqdm.auto import tqdm, trange
import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import ruamel.yaml

# My imports
from src.dataloader_weighting import WeightedNumpyDataset
from src.mnist.mnist_bin_vae import MnistBinVAE
from src.mnist.mnist_bin_rae import MnistBinRAE
from src import utils
from src.opt_scripts import base as wr_base
from src import GP_TRAIN_FILE, GP_OPT_FILE, DNGO_TRAIN_FILE


logger = logging.getLogger("mnist-complex-opt")


def setup_logger(logfile):
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def _run_command(command, command_name):
    logger.debug(f"{command_name} command:")
    logger.debug(command)
    start_time = time.time()
    run_result = subprocess.run(command, capture_output=True)
    assert run_result.returncode == 0, run_result.stderr
    logger.debug(f"{command_name} done in {time.time() - start_time:.1f}s")


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu):

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

    # Create trainer, default batch size: 32
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


def _batch_decode_z_and_props(model, z, args, get_props=True, filter_unique=False):
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
            return z_decode, z_prop
        else:
            return z_decode
    else:
        if get_props:
            return z_decode, z_prop
        else:
            return z_decode


def latent_optimization(args, model, datamodule, num_queries_to_do, bo_data_file, bo_run_folder, pbar=None, postfix=None):
    """
    perform latent space optimization using traditional local optimization strategies
    """
    ##################################################
    # Prepare BO
    ##################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(datamodule.data_train, dtype=torch.float32, device=device).unsqueeze(1)
    targets = datamodule.prop_train

    # Next, encode the data to latent space
    if args.gpu:
        model = model.cuda()
    latent_points = model.encode_to_latent_space(data)  # encode data to latent space
    model = model.cpu()  # Make sure to free up GPU memory
    torch.cuda.empty_cache()  # Free the memory up for tensorflow

    # Save points to file
    def _save_bo_data(latent_points, targets):

        # Prevent overfitting to bad points
        #targets = np.maximum(targets, args.invalid_score)
        targets = -targets.reshape(-1, 1)  # Since it is a minimization problem

        # Save the file
        np.savez_compressed(
        bo_data_file,
            X_train=latent_points.cpu().detach().numpy().astype(np.float64),
            X_test=[],
            y_train=targets.astype(np.float64),
            y_test=[],
        )

    _save_bo_data(latent_points, targets)
    
    # Part 1: fit surrogate model
    # ===============================
    iter_seed = int(np.random.randint(10000))
    
    new_bo_file = bo_run_folder / f"bo_train_res.npz"
    log_path = bo_run_folder / f"bo_train.log"

    # use sparse Gaussian process
    if args.bo_surrogate == "GP":
        gp_train_command = [
            "python",
            GP_TRAIN_FILE,
            f"--nZ={args.n_inducing_points}",
            f"--seed={iter_seed}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(new_bo_file)}",
            f"--logfile={str(log_path)}",
        ]

        # Add commands for initial fitting
        gp_fit_desc = "GP initial fit"
        gp_train_command += [
            "--init",
            "--kmeans_init",
        ]

        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        # Run command
        _run_command(gp_train_command, f"GP train")
        curr_bo_file = new_bo_file

    # use DNGO
    elif args.bo_surrogate == "DNGO":
        dngo_train_command = [
            "python",
            DNGO_TRAIN_FILE,
            f"--seed={iter_seed}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(new_bo_file)}",
            f"--logfile={str(log_path)}",
        ]

        # Add commands for initial fitting
        dngo_fit_desc = "DNGO initial fit"

        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(dngo_fit_desc)

        # Run command
        _run_command(dngo_train_command, f"DNGO train")
        curr_bo_file = new_bo_file
    else:
        raise NotImplementedError(args.bo_surrogate)

    # Part 2: optimize surrogate acquisition func to query point
    # ===============================

    # Run GP opt script
    opt_path = bo_run_folder / f"bo_opt_res.npy"
    log_path = bo_run_folder / f"bo_opt.log"
    if args.opt_constraint_threshold:
        bo_opt_command = [
            "python",
            GP_OPT_FILE,
            f"--seed={iter_seed}",
            f"--surrogate_file={str(curr_bo_file)}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(opt_path)}",
            f"--logfile={str(log_path)}",
            f"--n_samples={str(args.n_samples)}",
            f"--sample_distribution={str(args.sample_distribution)}",
            f"--n_out={str(num_queries_to_do)}",
            f"--bo_surrogate={args.bo_surrogate}",
            f"--n_starts={args.n_starts}",
            f"--opt_method={args.opt_method}",
            f"--pretrained_model_type={args.pretrained_model_type}",
            f"--opt_constraint_threshold={args.opt_constraint_threshold}",
            f"--opt_constraint_strategy={args.opt_constraint_strategy}",
            f"--n_gmm_components={args.n_gmm_components}",
        ]
    else:
        bo_opt_command = [
            "python",
            GP_OPT_FILE,
            f"--seed={iter_seed}",
            f"--surrogate_file={str(curr_bo_file)}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(opt_path)}",
            f"--logfile={str(log_path)}",
            f"--n_samples={str(args.n_samples)}",
            f"--sample_distribution={str(args.sample_distribution)}",
            f"--n_out={str(num_queries_to_do)}",
            f"--bo_surrogate={args.bo_surrogate}",
            f"--n_starts={args.n_starts}",
            f"--opt_method={args.opt_method}",
            f"--sparse_out={args.sparse_out}",
            f"--pretrained_model_type={args.pretrained_model_type}",
        ]
    if pbar is not None:
        pbar.set_description("optimizing acq func")
    _run_command(bo_opt_command, f"Surrogate opt")

    # Load point
    z_opt = np.load(opt_path)

    # Decode point
    x_new, y_new = _batch_decode_z_and_props(
        model,
        torch.as_tensor(z_opt, device=model.device),
        datamodule,
        args
    )

    # Reset pbar description
    if pbar is not None:
        pbar.set_description(old_desc)

        # Update best point in progress bar
        if postfix is not None:
            postfix["best"] = max(postfix["best"], float(max(y_new)))
            pbar.set_postfix(postfix)

    # Update datamodule with ALL data points
    return x_new, y_new, z_opt


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
        opt_point_properties=[],
        opt_latent_points=[],# saves corresponding function evaluations
        opt_model_version=[],
        latent_space_snapshots=[], # Uncomment for latent dimension = 2 to visualize latent manifold! saves corresponding retraining iteration
        params=str(sys.argv),
    )

    if model.latent_dim == 2:
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
        setup_logger(result_dir / "log.txt")

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

            if model.latent_dim == 2:
                latent_snapshot = _batch_decode_z_and_props(
                    model,
                    torch.as_tensor(results["latent_space_grid"], device=model.device),
                    args,
                    filter_unique=False,
                )[0]
                results["latent_space_snapshots"].append(latent_snapshot)

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )

            gp_dir = result_dir / "gp" / f"iter{samples_so_far}"
            gp_dir.mkdir(parents=True)
            bo_data_file = gp_dir / "data.npz"
            x_new, y_new, z_query = latent_optimization(
                args,
                model,
                datamodule,
                num_queries_to_do,
                bo_data_file=bo_data_file,
                bo_run_folder=gp_dir,
                pbar=pbar,
                postfix=postfix,
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
    parser = wr_base.add_gp_args(parser)

    args = parser.parse_args()

    main_loop(args)
