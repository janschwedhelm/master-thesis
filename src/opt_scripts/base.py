from argparse import ArgumentParser


parser = ArgumentParser()


def add_common_args(parser: ArgumentParser):
    opt_group = parser.add_argument_group("weighted retraining")
    opt_group.add_argument("--seed", type=int, required=True)
    opt_group.add_argument("--query_budget", type=int, required=True)
    opt_group.add_argument("--retraining_frequency", type=int, required=True)
    opt_group.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    opt_group.add_argument("--lso_strategy", type=str, default="opt")
    opt_group.add_argument("--result_root", type=str, required=True, help="root directory to store results in")
    opt_group.add_argument("--pretrained_model_file", type=str, default=None, help="path to pretrained model to use")
    opt_group.add_argument("--pretrained_model_type", type=str, default="vae")
    opt_group.add_argument("--pretrained_netg_model_file", type=str, default=None, help="path to pretrained generator model to use")
    opt_group.add_argument("--pretrained_netd_model_file", type=str, default=None, help="path to pretrained discriminator model to use")
    opt_group.add_argument("--pretrained_model_prior", type=str, default="normal")
    opt_group.add_argument("--n_retrain_steps", type=int, default=2222)
    opt_group.add_argument("--n_init_retrain_steps", type=int, default=222,
                           help="None to use n_retrain_steps, 0.0 to skip init retrain")
    opt_group.add_argument("--pretrained_predictor_file", type=str, default=None)
    opt_group.add_argument("--scaled_predictor_state_dict", type=str, default=None)
    opt_group.add_argument("--attr_file", type=str, default=None)
    opt_group.add_argument("--n_retrain_epochs", type=float, default=1.0)
    opt_group.add_argument("--n_init_retrain_epochs", type=float, default=None, help="None to use n_retrain_epochs, 0.0 to skip init retrain")

    return parser


def add_gp_args(parser: ArgumentParser):
    gp_group = parser.add_argument_group("Sparse GP")
    gp_group.add_argument("--n_inducing_points", type=int, default=500)
    gp_group.add_argument("--n_samples", type=int, default=10000)
    gp_group.add_argument("--n_starts",type=int,default=20,help="Number of optimization runs with different initial values")
    gp_group.add_argument("--n_out",type=int,default=5)
    gp_group.add_argument(
    "--opt_constraint_threshold",
    type=float,
    default=None,
    help="Threshold for optimization constraint"
    )
    gp_group.add_argument(
        "--opt_constraint_strategy",
        type=str,
        default="gmm_fit"
    )
    gp_group.add_argument(
        "--n_gmm_components",
        type=int,
        default=None,
        help="Number of components used for GMM fitting"
    )
    gp_group.add_argument(
        "--sparse_out",
        type=bool,
        default=True,
    )
    gp_group.add_argument(
        "--opt_method",
        type=str,
        default="L-BFGS-B",
    )
    gp_group.add_argument(
        "--bo_surrogate",
        type=str,
        default="GP",
    )
    gp_group.add_argument("--n_rand_points", type=int, default=8000)
    gp_group.add_argument("--n_best_points", type=int, default=2000)
    gp_group.add_argument("--sample_distribution", type=str, default="normal")
    gp_group.add_argument("--ground_truth_discriminator_threshold", type=float, default=0.0)
    gp_group.add_argument("--M", type=int, default=10)
    gp_group.add_argument("--M_0", type=int, default=100)
    gp_group.add_argument("--N", type=int, default=1000)
    return parser
