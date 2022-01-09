import numpy as np
import pytorch_lightning as pl
import argparse
import pickle
import time
from pathlib import Path

from src.fid.fid_score import calculate_fid_given_tensors
from src.dataloader_celeba_weighting import CelebaWeightedTensorDataset
from src import utils

NUM_WORKERS = 2
DIMS = 2048


def main(args):
    # Seeding
    pl.seed_everything(args.seed)

    device = 'cuda:0' if args.gpu else 'cpu'

    # Make result directory
    result_dir = Path(args.result_dir).resolve()

    # Load pre-trained model
    #if args.pretrained_model_type == "vae":
    #    model = MnistBinVAE.load_from_checkpoint(args.pretrained_model_file)
    #    model.beta = model.hparams.beta_final  # Override any beta annealing
    #elif args.pretrained_model_type == "rae":
    #    model = MnistBinRAE.load_from_checkpoint(args.pretrained_model_file)
    #else:
    #    raise NotImplementedError(args.pretrained_model_type)

    # Load data
    #datamodule = WeightedNumpyDataset(args, utils.DataWeighter(args))
    #datamodule.setup("fit")

    with np.load(args.sample_path) as npz:
        samples = npz["opt_points"]

    datamodule = CelebaWeightedTensorDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    print("Start FID computation")

    start_time = time.time()
    fid_score = calculate_fid_given_tensors(samples, dataloader, args.batch_size,
                                            device, dims=DIMS, num_workers=NUM_WORKERS)
    time_needed = time.time() - start_time
    results = dict(test_set=args.sample_path,
                   fid_score=fid_score,
                   time=time_needed)

    print(f"Resulting FID score: {fid_score}")
    print(f"Time needed: {time_needed} seconds")

    # if "normal" in args.sampling_method:
    #     print("Start 'normal' sampling")
    #     samples = latent_sampling(args, model, datamodule, args.n_samples, sampling_method="normal",
    #                               get_props=False, filter_unique=False)
    #     if args.test_set == "validation":
    #         print("Start 'normal' FID computation")
    #         fid_score = calculate_fid_given_arrays(samples, datamodule.data_val, args.batch_size,
    #                                                device, dims=DIMS, num_workers=NUM_WORKERS)
    #         results['normal'] = dict(test_set="validation",
    #                                  fid_score=fid_score)
    #         print(f"Resulting FID score with sampling method 'normal': {fid_score}")
    #     elif args.test_set == "train":
    #         print("Start 'normal' FID computation")
    #         fid_score = calculate_fid_given_arrays(samples, datamodule.data_train, args.batch_size,
    #                                                device, dims=DIMS, num_workers=NUM_WORKERS)
    #         results['normal'] = dict(test_set="train",
    #                                  fid_score=fid_score)
    #         print(f"Resulting FID score with sampling method 'normal': {fid_score}")
    #     elif args.test_set.endswith(".npz"):
    #         print("Start 'normal' FID computation")
    #         with np.load(args.test_set) as npz:
    #             test_samples = npz["data"]
    #         fid_score = calculate_fid_given_arrays(samples, test_samples, args.batch_size,
    #                                                device, dims=DIMS, num_workers=NUM_WORKERS)
    #         results['normal'] = dict(test_set=args.test_set,
    #                                  fid_score=fid_score)
    #         print(f"Resulting FID score with sampling method 'normal': {fid_score}")
    #     else:
    #         raise NotImplementedError(args.test_set)
    # if "gmm" in args.sampling_method:
    #     print("Start 'gmm' sampling")
    #     samples = latent_sampling(args, model, datamodule, args.n_samples, sampling_method="gmm",
    #                               get_props=False, filter_unique=False)
    #     if args.test_set == "validation":
    #         print("Start 'gmm' FID computation")
    #         fid_score = calculate_fid_given_arrays(samples, datamodule.data_val, args.batch_size,
    #                                                device, dims=DIMS, num_workers=NUM_WORKERS)
    #         results['gmm'] = dict(test_set="validation",
    #                               n_components=args.n_gmm_components,
    #                               fid_score=fid_score)
    #         print(f"Resulting FID score with sampling method 'gmm': {fid_score}")
    #     elif args.test_set == "train":
    #         print("Start 'normal' FID computation")
    #         fid_score = calculate_fid_given_arrays(samples, datamodule.data_train, args.batch_size,
    #                                                device, dims=DIMS, num_workers=NUM_WORKERS)
    #         results['gmm'] = dict(test_set="train",
    #                               fid_score=fid_score)
    #         print(f"Resulting FID score with sampling method 'normal': {fid_score}")
    #     elif args.test_set.endswith(".npz"):
    #         print("Start 'normal' FID computation")
    #         with np.load(args.test_set) as npz:
    #             test_samples = npz["data"]
    #         fid_score = calculate_fid_given_arrays(samples, test_samples, args.batch_size,
    #                                                device, dims=DIMS, num_workers=NUM_WORKERS)
    #         results['gmm'] = dict(test_set=args.test_set,
    #                               fid_score=fid_score)
    #         print(f"Resulting FID score with sampling method 'normal': {fid_score}")
    #     else:
    #         raise NotImplementedError(args.test_set)
    #
    with open(result_dir, "wb") as f:
        pickle.dump(results, f)

    print("Computation finished; end of script")


if __name__ == "__main__":
    # arguments and argument checking
    parser = argparse.ArgumentParser()

    parser = CelebaWeightedTensorDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)

    # Add FID score specific arguments
    fid_group = parser.add_argument_group(title="fid")
    fid_group.add_argument("--seed", type=int, required=True)
    #fid_group.add_argument(
    #    "--sampling_method",
    #    nargs="+",
    #    default=["normal", "gmm"],
    #    help="Can be either 'normal' or 'gmm' or both, wrapped in a list: ['normal','gmm']",
    #)
    #fid_group.add_argument("--n_gmm_components", type=int, default=10)
    #fid_group.add_argument(
    #    "--n_samples",
    #    type=int,
    #    default=10000,
    #    help="Number of samples to draw for FID computation",
    #)
    fid_group.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    fid_group.add_argument("--result_dir", type=str, required=True, help="directory to store results in")
    #fid_group.add_argument("--pretrained_model_file", type=str, required=True, help="path to pretrained model to use")
    #fid_group.add_argument("--pretrained_model_type", type=str, default="vae")
    fid_group.add_argument("--sample_path", type=str, required=True, help="a .npz"
                                                                          "file with 'opt_points' key containing the test samples")

    args = parser.parse_args()

    main(args)