"""
Makes an MNIST dataset with specific target property
"""

import numpy as np
import argparse
from pathlib import Path

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir", type=str, required=True, help="directory to save files in"
)
parser.add_argument(
    "--mnist_dir", type=str, help="directory with MNIST datafile (.npz file)", default="data/mnist/mnist.npz"
)
parser.add_argument(
    "--binary", type=bool, help="if True, images are binarized", default=False
)


if __name__ == "__main__":

    args = parser.parse_args()
    assert Path(args.save_dir).exists()
    assert Path(args.mnist_dir).exists()

    # Load full MNIST dataset
    with np.load(args.mnist_dir) as npz:
        x_train = npz["x_train"]
        y_train = npz["y_train"]
        x_test = npz["x_test"]
        y_test = npz["y_test"]

    # Binarize dataset if specified
    if args.binary:
        x_train = np.array(x_train > 0, dtype=np.float32)
        x_test = np.array(x_test > 0, dtype=np.float32)
    else:
        x_train = x_train.astype('float32')
        x_train = x_train / 255.0
        x_test = x_test.astype('float32')
        x_test = x_test / 255.0


    # Concatenate training and test data as we do not need the split
    data = np.concatenate([x_train, x_test])
    targets = np.concatenate([y_train.astype("int"), y_test.astype("int")])

    # Prepare for saving dataset dataset
    shapes_desc = "mnist"
    save_name = f"{shapes_desc}_B{args.binary}"

    np.savez_compressed(
        str(Path(args.save_dir) / save_name) + ".npz", data=data, targets=targets
    )

    print(f"Dataset created. Total of {len(targets)} points")
    print(f"Array size {data.nbytes / 1e9:.1f} Gb")
