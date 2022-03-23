"""
Makes an MNIST dataset with specific target property
"""

import numpy as np
import argparse
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import pickle
import json

from src.utils import rounddown


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir", type=str, required=True, help="directory to save files in"
)
parser.add_argument(
    "--celeba_dir", type=str, help="directory with Celeba datafiles (.jpg files)", default="data/celeba/faces"
)
#parser.add_argument(
#    "--attr_path", type=str
#)

if __name__ == "__main__":

    args = parser.parse_args()
    assert Path(args.save_dir).exists()
    assert Path(args.celeba_dir).exists()

    #lines = [line.rstrip() for line in open(args.attr_path, 'r')]
    image_filename_set = set(os.listdir(args.celeba_dir))
    transform = transforms.Compose([transforms.ToTensor()])

    for i, filename in enumerate(image_filename_set):
        if i % 5000 == 0:
            print(i)
        #split = line.split()
        #filename = split[0]
        #if filename in image_filename_set:

        filename_idx = int(filename.split('.')[0])
        upper_level = rounddown(filename_idx, 1000)
        middle_level = rounddown(filename_idx, 100)
        lower_level = rounddown(filename_idx, 10)

        if not os.path.exists(str(Path(args.save_dir) / f"{upper_level}" / f"{middle_level}" / f"{lower_level}")):
            os.makedirs(str(Path(args.save_dir) / f"{upper_level}" / f"{middle_level}" / f"{lower_level}"))

        image = Image.open(os.path.join(args.celeba_dir, filename)).convert('RGB')
        image = transform(image)

        torch.save(image, args.save_dir + f"/{upper_level}/{middle_level}/{lower_level}/{filename_idx}.pt", pickle_protocol=pickle.HIGHEST_PROTOCOL)
