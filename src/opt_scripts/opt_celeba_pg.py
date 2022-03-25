""" Run weighted retraining for CelebA with the SN-GAN model """

import argparse
from pathlib import Path
import os
import sys
import logging
import subprocess
import time
from tqdm.auto import tqdm
import ruamel.yaml

from src.dataloader_celeba_weighting import *
from src.temperature_scaling import *
from src.resnet50 import resnet50
from src import utils
from src.opt_scripts import base as wr_base
from src.torch_mimicry.training.trainer import Trainer
from src.projected_gan.pg_modules.networks_fastgan import Generator
from src.projected_gan.pg_modules.discriminator import ProjectedDiscriminator
from src.dataloader_celeba_weighting import CelebaWeightedTensorDataset
from src import GP_TRAIN_FILE, DNGO_TRAIN_FILE, GP_OPT_SAMPLING_FILE
from src.projected_gan.train import train

logger = logging.getLogger("celeba-opt")


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


def retrain_model(args, G, D, G_ema, dataloader, save_dir, version_str, num_steps):
    train(G=G, D=D, G_ema=G_ema, num_steps=num_steps, dataloader=dataloader, outdir=str(Path(save_dir) / version_str),
          num_gpus=1, batch_size=args.batch_size, batch_gpu=args.batch_size,
          tick=1000, snap=1000000, seed=args.seed, restart_every=9999999, ema_num_steps=int(num_steps/100))



def generate_samples(netG, netD, n_samples, discriminator_quantile_threshold, opt_method, device):
    """ helper function to generate samples """
    batch_size = 50
    images = torch.zeros(size=(n_samples, 3, 64, 64))
    if opt_method != "sampling":
        latents = torch.zeros(size=(n_samples, 128))
    else:
        latents = torch.zeros(size=(n_samples, 64))
    seed = int(np.random.randint(10000))
    pl.seed_everything(seed)

    with torch.no_grad():
        # Set model to evaluation mode
        netG.eval()

        # Inference variables
        batch_size = min(n_samples, batch_size)

        # Collect all samples
        for idx in range(n_samples // batch_size):
            # Collect fake image
            fake_images, z = netG.generate_images(num_images=batch_size, device=device)
            fake_images = fake_images.cpu()
            z = z.cpu()
            images[idx*batch_size : idx*batch_size + batch_size] = fake_images
            latents[idx*batch_size : idx*batch_size + batch_size] = z
            del fake_images
            del z

    if discriminator_quantile_threshold != 0.0:
        with torch.no_grad():
            # Set model to evaluation mode
            netD.eval()
    
            # Inference variables
            batch_size = min(n_samples, batch_size)
    
            # Collect all samples()
            probabilities = []
            start_time = time.time()
            for idx in range(0, n_samples, batch_size):
                # Collect fake image
                probas = torch.sigmoid(netD(images[idx: idx + batch_size].to(device)))
                probas = probas.cpu().numpy()
                probabilities.append(probas)
    
            probabilities = np.concatenate(probabilities, axis=0)
            probabilities = probabilities.flatten()
    
            threshold_probability = np.quantile(probabilities, discriminator_quantile_threshold)
            good_idx = np.where(probabilities >= threshold_probability)
    
            return images[good_idx], latents[good_idx], threshold_probability
    else:
        return images, latents, None


def _choose_best_rand_points(latents, targets):
    """ helper function to choose points for training surrogate model """
    chosen_point_set = set()

    # Best scores at start
    targets_argsort = np.argsort(-targets.flatten())
    for i in range(args.n_best_points):
        chosen_point_set.add(targets_argsort[i])
    candidate_rand_points = np.random.choice(
        len(targets_argsort),
        size=args.n_rand_points + args.n_best_points,
        replace=False,
    )
    for i in candidate_rand_points:
        if i not in chosen_point_set and len(chosen_point_set) < (
            args.n_rand_points + args.n_best_points
        ):
            chosen_point_set.add(i)
    assert len(chosen_point_set) == (args.n_rand_points + args.n_best_points)

    chosen_latents = latents[list(chosen_point_set)]
    chosen_targets = targets[list(chosen_point_set)]

    return chosen_latents.detach().numpy(), chosen_targets


def _batch_get_props(images, predictor, device):
    """ helper function to obtain target properties from images """
    batch_size = 50

    with torch.no_grad():
        # Inference variables
        batch_size = min(images.shape[0], batch_size)
        
        properties = []
        for idx in range(0, images.shape[0], batch_size):
            # normalize images
            images_upscaled = torch.nn.functional.interpolate(images[idx : idx + batch_size].to(device), size=(128, 128), mode='bicubic',
                                                              align_corners=False)
            img_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            img_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            images_normalized = (images_upscaled - img_mean) / img_std
        
            predictions = predictor(images_normalized)
            probas_predictions = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()
            props = probas_predictions @ np.array([0, 1, 2, 3, 4, 5])
            properties.append(props)
            del images_normalized
    
    properties = np.concatenate(properties, axis=0)

    return properties


def _batch_decode_z_and_props(netG, predictor, z, get_props=True):
    """
    helper function to decode some latent vectors and calculate their properties
    """
    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 50
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = netG(z[j: j + batch_size])
            z_decode.append(img.cpu())
            del img
    
    # Concatentate all points and convert to numpy
    z_decode = torch.cat(z_decode, dim=0).to(netG.device)

    z_prop = _batch_get_props(z_decode, predictor, netG.device)

    if get_props:
        return z_decode, z_prop
    else:
        return z_decode


def latent_optimization(args, netG, netD, scaled_predictor, dataloader, num_queries_to_do, bo_data_file, bo_run_folder, device, pbar=None, postfix=None, curr_generator_file=None, curr_discriminator_file=None):
    """ perform latent space optimization using SN-GANs """

    ##################################################
    # Prepare BO
    ##################################################
    
    ground_truth_images, ground_truth_latents, _ = generate_samples(netG, netD, args.n_samples,
                                                                    args.ground_truth_discriminator_threshold, args.opt_method, device)
                                                                    
    pred_targets = _batch_get_props(ground_truth_images, scaled_predictor, device)
    
    latent_points, targets = _choose_best_rand_points(ground_truth_latents, pred_targets)

    netG = netG.cpu()  # Make sure to free up GPU memory
    netD = netD.cpu()
    torch.cuda.empty_cache()  # Free the memory up for tensorflow

    # Save points to file
    def _save_bo_data(latent_points, targets):

        # Prevent overfitting to bad points
        targets = -targets.reshape(-1, 1)  # Since it is a minimization problem

        # Save the file
        np.savez_compressed(
        bo_data_file,
            X_train=latent_points.astype(np.float64),
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
        print("Start training DNGO")
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
        if args.n_gmm_components:
            bo_opt_command = [
                "python",
                GP_OPT_SAMPLING_FILE,
                f"--seed={iter_seed}",
                f"--surrogate_file={str(curr_bo_file)}",
                f"--data_file={str(bo_data_file)}",
                f"--save_file={str(opt_path)}",
                f"--logfile={str(log_path)}",
                f"--n_samples={args.n_samples}",
                f"--sample_distribution={str(args.sample_distribution)}",
                f"--n_out={str(num_queries_to_do)}",
                f"--bo_surrogate={args.bo_surrogate}",
                f"--n_starts={args.n_starts}",
                f"--opt_method={args.opt_method}",
                f"--sparse_out={args.sparse_out}",
                f"--pretrained_model_prior={args.pretrained_model_prior}",
                f"--opt_constraint_threshold={args.opt_constraint_threshold}",
                f"--opt_constraint_strategy={args.opt_constraint_strategy}",
                f"--n_gmm_components={args.n_gmm_components}",
            ]
        else:
            bo_opt_command = [
                "python",
                GP_OPT_SAMPLING_FILE,
                f"--seed={iter_seed}",
                f"--surrogate_file={str(curr_bo_file)}",
                f"--data_file={str(bo_data_file)}",
                f"--save_file={str(opt_path)}",
                f"--logfile={str(log_path)}",
                f"--n_samples={args.n_samples}",
                f"--sample_distribution={args.sample_distribution}",
                f"--n_out={num_queries_to_do}",
                f"--bo_surrogate={args.bo_surrogate}",
                f"--n_starts={args.n_starts}",
                f"--opt_method={args.opt_method}",
                f"--sparse_out={args.sparse_out}",
                f"--pretrained_model_prior={args.pretrained_model_prior}",
                f"--opt_constraint_threshold={args.opt_constraint_threshold}",
                f"--opt_constraint_strategy={args.opt_constraint_strategy}",
                f"--curr_generator_file={str(curr_generator_file)}",
                f"--curr_discriminator_file={str(curr_discriminator_file)}"
            ]
    else:
        bo_opt_command = [
            "python",
            GP_OPT_SAMPLING_FILE,
            f"--seed={iter_seed}",
            f"--surrogate_file={str(curr_bo_file)}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(opt_path)}",
            f"--logfile={str(log_path)}",
            f"--n_samples={args.n_samples}",
            f"--sample_distribution={str(args.sample_distribution)}",
            f"--n_out={str(num_queries_to_do)}",
            f"--bo_surrogate={args.bo_surrogate}",
            f"--n_starts={args.n_starts}",
            f"--opt_method={args.opt_method}",
            f"--sparse_out={args.sparse_out}",
            f"--opt_method={args.opt_method}",
            f"--pretrained_model_prior={args.pretrained_model_prior}",
        ]
    if pbar is not None:
        pbar.set_description("optimizing acq func")
    print("Start optimizing DNGO")
    _run_command(bo_opt_command, f"Surrogate opt")

    # Load point
    z_opt = np.load(opt_path)
    
    # Decode point
    netG.to(device)
    x_new, y_new = _batch_decode_z_and_props(
        netG,
        scaled_predictor,
        torch.as_tensor(z_opt, device=device)
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

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Load data
    datamodule = CelebaWeightedTensorDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    # Load pre-trained SN-GAN generator / discriminator
    G = Generator().to(device)
    G_ema = Generator().to(device)

    D = ProjectedDiscriminator(backbone_kwargs={'num_discs': 4}).to(device)

    G.load_state_dict(torch.load(args.pretrained_g_model_file))
    G_ema.load_state_dict(torch.load(args.pretrained_g_ema_model_file))
    D.load_state_dict(torch.load(args.pretrained_d_model_file))
    #optD = optim.Adam(D.parameters(), 1e-8, betas=(0.0, 0.99))
    #optG = optim.Adam(G.parameters(), 1e-8, betas=(0.0, 0.99))

    # Load pretrained (temperature-scaled) CelebA-Dialog predictor
    checkpoint_predictor = torch.load(args.pretrained_predictor_file)
    predictor = resnet50(attr_file=args.attr_file)
    predictor.load_state_dict(checkpoint_predictor['state_dict'], strict=True)
    predictor.eval()

    state_dict_scaled_predictor = torch.load(args.scaled_predictor_state_dict)
    scaled_predictor = ModelWithTemperature(predictor, args.property_id)
    scaled_predictor.load_state_dict(state_dict_scaled_predictor, strict=True)
    scaled_predictor.to(device)
    scaled_predictor.eval()

    # Set up results tracking
    results = dict(
        opt_points=[],  # saves (default: 5) optimal points in the original input space for each retraining iteration
        opt_point_properties=[], # saves corresponding function evaluations
        opt_latent_points=[], # saves corresponding latent points
        opt_model_version=[],
        params=str(sys.argv),
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
            num_steps = args.n_retrain_steps
            if ret_idx == 0 and args.n_init_retrain_steps is not None:
                # default: initial fine-tuning of pre-trained model for 1 epoch
                num_steps = args.n_init_retrain_steps
            if num_steps > 0:
                retrain_dir = result_dir / "retraining"
                version = f"retrain_{samples_so_far}"
                # default: run through 10% of the weighted training data in retraining epoch
                retrain_model(
                    args, G, D, G_ema, dataloader, retrain_dir, version, num_steps
                )

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            ########### Continue here!
            if args.lso_strategy == "opt":
                gp_dir = result_dir / "gp" / f"iter{samples_so_far}"
                curr_discriminator_file = result_dir / "retraining" / f"retrain_{samples_so_far}" / "checkpoints" / "netD" / f"netD_{num_steps}_steps.pth"
                curr_generator_file = result_dir / "retraining" / f"retrain_{samples_so_far}" / "checkpoints" / "netG" / f"netG_{num_steps}_steps.pth"
                if os.path.exists(result_dir / "retraining" / f"retrain_{samples_so_far}" / "checkpoints" / "netD" / f"netD_{num_steps}_steps.pth"):
                    curr_discriminator_file = result_dir / "retraining" / f"retrain_{samples_so_far}" / "checkpoints" / "netD" / f"netD_{num_steps}_steps.pth"
                    curr_generator_file = result_dir / "retraining" / f"retrain_{samples_so_far}" / "checkpoints" / "netG" / f"netG_{num_steps}_steps.pth"
                else:
                    curr_discriminator_file = None
                    curr_generator_file = None
                    
                gp_dir.mkdir(parents=True)
                bo_data_file = gp_dir / "data.npz"
                x_new, y_new, z_query = latent_optimization(
                    args,
                    G,
                    D,
                    scaled_predictor,
                    dataloader,
                    num_queries_to_do,
                    bo_data_file=bo_data_file,
                    bo_run_folder=gp_dir,
                    device=device,
                    pbar=pbar,
                    postfix=postfix,
                    curr_generator_file = curr_generator_file,
                    curr_discriminator_file = curr_discriminator_file
                )
            else:
                raise NotImplementedError(args.lso_strategy)

            # Save new tensor data
            if not os.path.exists(str(Path(data_dir) / f"sampled_data_iter{samples_so_far}")):
                os.makedirs(str(Path(data_dir) / f"sampled_data_iter{samples_so_far}"))

            new_filename_list = []
            for i, x in enumerate(x_new):
                torch.save(x, str(Path(data_dir) / f"sampled_data_iter{samples_so_far}/tensor_{i}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
                new_filename_list.append(str(Path(data_dir) / f"sampled_data_iter{samples_so_far}/tensor_{i}.pt"))
            
            # Append new points to dataset and adapt weighting
            datamodule.append_train_data(new_filename_list, y_new)

            # Save results
            results["opt_latent_points"] += [z for z in z_query]
            results["opt_points"] += [x.detach().cpu().numpy() for x in x_new]
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
    parser = CelebaWeightedTensorDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    parser = wr_base.add_common_args(parser)
    parser = wr_base.add_gp_args(parser)

    args = parser.parse_args()

    main_loop(args)
