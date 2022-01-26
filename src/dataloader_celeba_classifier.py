from torch.utils import data
import torch
import pickle
import random
import numpy as np

from src.utils import rounddown


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, filename_set_path, data_dir, attr_path, mode, attribute_id=None, max_property_value=5,
                 min_property_value=0):
        """Initialize and preprocess the CelebA dataset."""
        self.data_dir = data_dir
        self.filename_set_path = filename_set_path
        self.attr_path = attr_path
        self.mode = mode
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.full_dataset = []
        self.attribute_id = attribute_id
        self.max_property_value = max_property_value
        self.min_property_value = min_property_value

        self.preprocess()

        if self.mode == 'train':
            self.num_images = len(self.train_dataset)
        elif self.mode == 'val':
            self.num_images = len(self.val_dataset)
        elif self.mode == 'all':
            self.num_images = len(self.full_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        if self.mode != 'all':
            lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        else:
            lines = [line.rstrip() for line in open(self.attr_path, 'r')][1:]
        with open(self.filename_set_path, "rb") as f:
            filename_set = pickle.load(f)

        random.seed(1234)
        random.shuffle(lines)
        print("Start preprocessing")
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            targets = np.array([float(i) for i in split[1:]])
            if self.attribute_id is not None:
                targets = targets[self.attribute_id]

            if filename in filename_set:
                if self.mode == "train":
                    if self.attribute_id is None or (targets <= self.max_property_value) & (targets >= self.min_property_value):
                        self.train_dataset.append([filename, targets])
                if self.mode == "val":
                    if self.attribute_id is None or (targets <= self.max_property_value) & (targets >= self.min_property_value):
                        self.val_dataset.append([filename, targets])
                if self.mode == "all":
                    if self.attribute_id is None or (targets <= self.max_property_value) & (targets >= self.min_property_value):
                        self.full_dataset.append([filename, targets])
                else:
                    if self.attribute_id is None or (targets <= self.max_property_value) & (targets >= self.min_property_value):
                        self.test_dataset.append([filename, targets])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        if self.mode == "train":
            dataset = self.train_dataset
        if self.mode == "val":
            dataset = self.val_dataset
        if self.mode == "all":
            dataset = self.full_dataset
        else:
            dataset = self.test_dataset

        filename, targets = dataset[index]
        filename_idx = int(filename.split('.')[0])
        upper_level = rounddown(filename_idx, 1000)
        middle_level = rounddown(filename_idx, 100)
        lower_level = rounddown(filename_idx, 10)

        image = torch.load(self.data_dir + f"/{upper_level}/{middle_level}/{lower_level}/{filename_idx}.pt").unsqueeze(0)

        return image.squeeze(0), targets

    def __len__(self):
        """Return the number of images."""
        return self.num_images
