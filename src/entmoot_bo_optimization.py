""" Code to train tree-based model as surrogate model and to deterministically solve resulting MIO """

from entmoot.optimizer.optimizer import Optimizer
from src.celeba_obj_function import *

import copy
import inspect
import numbers
import logging
import numpy as np
import time
import argparse
import pytorch_lightning as pl


try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--logfile",
    type=str,
    help="file to log to",
    default="bo_train_opt.log"
)
parser.add_argument(
    "--seed",
    type=int,
    required=True
)
parser.add_argument(
    "--data_file",
    type=str,
    help="file to load data from",
    required=True
)
parser.add_argument(
    "--save_file",
    type=str,
    required=True,
    help="File to save results to"
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True
)
parser.add_argument(
    "--n_out",
    type=int,
    default=5,
    help="Number of optimization points to return"
)



def entmoot_train_opt(
        data_file,
        save_file,
        func,
        dimensions,
        logfile="entmoot_train.log",
        n_calls=5,
        batch_strategy="cl_mean",
        n_initial_points=0,
        batch_size=None,
        base_estimator="GBRT",
        std_estimator=None,
        initial_point_generator="random",
        acq_func="LCB",
        acq_optimizer="global",
        random_state=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
        base_estimator_kwargs={'n_estimators': 800, 'min_child_samples': 20, 'max_depth': 2, 'num_leaves': 5},
        std_estimator_kwargs=None,
        model_queue_size=None,
        verbose=1,
):
    """ function to perform deterministic optimization with trained tree-based model """
    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    if acq_optimizer_kwargs is None:
        acq_optimizer_kwargs = {}

    # Initialize optimization
    # Suppose there are points provided (x0 and y0), record them
    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        x0 = npz['X_train'].tolist()
        y0 = npz['y_train'].ravel().tolist()
    
    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    if n_initial_points <= 0 and not x0:
        raise ValueError("Either set `n_initial_points` > 0,"
                         " or provide `x0`")
    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    required_calls = n_initial_points + (len(x0) if not y0 else 0)
    if n_calls < required_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))
    # calculate the total number of initial points
    n_initial_points = n_initial_points + len(x0)

    # Build optimizer

    # create optimizer class
    optimizer = Optimizer(
        dimensions,
        base_estimator=base_estimator,
        std_estimator=std_estimator,
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        acq_func=acq_func,
        acq_optimizer=acq_optimizer,
        random_state=random_state,
        acq_func_kwargs=acq_func_kwargs,
        acq_optimizer_kwargs=acq_optimizer_kwargs,
        base_estimator_kwargs=base_estimator_kwargs,
        std_estimator_kwargs=std_estimator_kwargs,
        model_queue_size=model_queue_size,
        verbose=verbose
    )

    # Record provided points

    # create return object
    result = None
    # evaluate y0 if only x0 is provided
    if x0 and y0 is None:
        y0 = list(map(func, x0))
        n_calls -= len(y0)
    # record through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        LOGGER.info("Start initial model fitting")
        start_time = time.time()

        result = optimizer.tell(x0, y0)
        end_time = time.time()
        LOGGER.info(f"Initial model fitting took {end_time - start_time:.1f}s to finish")
        result.specs = specs

    # Handle solver output
    if not isinstance(verbose, (int, type(None))):
        raise TypeError("verbose should be an int of [0,1,2] or bool, "
                        "got {}".format(type(verbose)))

    if isinstance(verbose, bool):
        if verbose:
            verbose = 1
        else:
            verbose = 0
    elif isinstance(verbose, int):
        if verbose not in [0, 1, 2]:
            raise TypeError("if verbose is int, it should in [0,1,2], "
                            "got {}".format(verbose))

    # Optimize
    _n_calls = n_calls

    itr = 1

    if verbose > 0:
        LOGGER.info("")
        LOGGER.info("SOLVER: start solution process...")
        LOGGER.info("")
        LOGGER.info(f"SOLVER: generate \033[1m {n_initial_points}\033[0m initial points...")

    while _n_calls > 0:

        # check if optimization is performed in batches
        if batch_size is not None:
            _batch_size = min([_n_calls, batch_size])
            _n_calls -= _batch_size
            next_x = optimizer.ask(_batch_size, strategy=batch_strategy)
        else:
            _n_calls -= 1
            logging.info(f"Start optimization of acquisition function for iteration {itr}")
            start_time = time.time()
            next_x = optimizer.ask(strategy=batch_strategy)
            end_time = time.time()
            LOGGER.info(f"Optimization procedure for iteration {itr} took {end_time - start_time:.1f}s to finish")
            LOGGER.info(f"Suggested sample: {next_x}")

        next_y = func(next_x)
        best_fun = result.fun

        # handle output print at every iteration
        if verbose > 0:
            LOGGER.info("")
            LOGGER.info(f"itr_{itr}")

            if isinstance(next_y, Iterable):
                # in case of batch optimization, print all new proposals and
                # mark improvements of objectives with (*)
                print_str = []
                for y in next_y:
                    if y <= best_fun:
                        print_str.append(f"\033[1m{round(y, 5)}\033[0m (*)")
                    else:
                        print_str.append(str(round(y, 5)))
                LOGGER.info(f"   new points obj.: {print_str[0]}")
                for y_str in print_str[1:]:
                    LOGGER.info(f"                    {y_str}")
            else:
                # in case of single point sequential optimization, print new
                # point proposal
                if next_y <= best_fun:
                    print_str = f"\033[1m itr_{round(next_y, 5)}\033[0m"
                else:
                    print_str = str(round(next_y, 5))
                LOGGER.info(f"   new point obj.: {round(next_y, 5)}")

            # print best obj until (not including) current iteration
            LOGGER.info(f"   best obj.:       {round(best_fun, 5)}")

        itr += 1
        
        if n_calls > 0:
            LOGGER.info("Refit model")
        start_time = time.time()
        result = optimizer.tell(
            next_x, next_y,
            fit=batch_size is None and not _n_calls <= 0
        )
        end_time = time.time()
        if n_calls > 0:
            LOGGER.info(f"Refitting model took {end_time - start_time:.1f}s to finish")

        result.specs = specs

    # print end of solve once convergence criteria is met
    if verbose > 0:
        LOGGER.info("")
        LOGGER.info("SOLVER: finished retraining iteration!")
        LOGGER.info(f"SOLVER: best obj.: {round(result.fun, 5)}")
        LOGGER.info("")
    
    # Save results
    latent_pred = np.array(result.x_iters[-n_calls:], dtype=np.int64)
    np.save(save_file, latent_pred)
    
    # Make some gp predictions in the log file
    LOGGER.info("Acquisition function results")
    LOGGER.info("mu at points:")
    LOGGER.info(list(np.array(result.model_mu[-5:]).ravel()))

    LOGGER.info("\n\nEND OF SCRIPT!")
    
    return latent_pred


if __name__ == "__main__":

    
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    func = CelebaTargetPredictor(args.model_path)
    
    entmoot_train_opt(
        args.data_file,
        args.save_file,
        func,
        func.get_bounds(),
        logfile=args.logfile,
        n_calls=args.n_out,
        batch_strategy="cl_mean",
        n_initial_points=0,
        batch_size=None,
        base_estimator="GBRT",
        std_estimator=None,
        initial_point_generator="random",
        acq_func="LCB",
        acq_optimizer="global",
        random_state=1,
        acq_func_kwargs=None,
        acq_optimizer_kwargs={'gurobi_timelimit': 2*60},
        base_estimator_kwargs={'n_estimators': 800, 'min_child_samples': 20, 'max_depth': 2, 'num_leaves': 5, 'device': 'cpu'},
        std_estimator_kwargs=None,
        model_queue_size=None,
        verbose=1,
    )
    