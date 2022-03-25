from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import pytorch_lightning as pl

from src.dataloader_celeba_classifier import *

NUM_WORKERS = 4


class CelebaWeightedTensorDataset(pl.LightningDataModule):
    """ Implements a weighted tensor dataset (used for CelebA task) """

    def __init__(self, hparams, data_weighter):
        super().__init__()
        self.tensor_dir = hparams.tensor_dir
        self.property_id = hparams.property_id
        self.max_property_value = hparams.max_property_value
        self.batch_size = hparams.batch_size
        self.train_attr_path = hparams.train_attr_path
        self.val_attr_path = hparams.val_attr_path
        self.combined_annotation_path = hparams.combined_annotation_path
        self.filename_set_path = hparams.filename_set_path

        self.mode = hparams.mode

        self.data_weighter = data_weighter

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument(
            "--tensor_dir", type=str, required=True, help="path to folder of tensor files"
        )

        data_group.add_argument("--batch_size", type=int, default=128)
        data_group.add_argument(
            "--property_id",
            type=int,
            required=True,
            help="Attribute id of object property",
        )
        data_group.add_argument(
            "--max_property_value",
            type=int,
            default=2,
        )
        data_group.add_argument(
            "--train_attr_path", type=str,
            required=True,
        )
        data_group.add_argument(
            "--val_attr_path", type=str,
            required=True,
        )
        data_group.add_argument(
            "--combined_annotation_path", type=str,
            required=True,
        )
        data_group.add_argument(
            "--filename_set_path", type=str,
            required=True,
        )
        data_group.add_argument(
            "--mode", type=str,
            default="split",
        )
        return parent_parser

    def prepare_data(self):
        pass

    def setup(self, stage):
        if self.mode == "split":
            train_dataset = CelebA(self.filename_set_path, self.tensor_dir, self.train_attr_path, mode='train',
                             attribute_id=self.property_id, max_property_value=self.max_property_value)
            val_dataset = CelebA(self.filename_set_path, self.tensor_dir, self.val_attr_path, mode='val',
                             attribute_id=self.property_id, max_property_value=self.max_property_value)

            train_dataset_as_numpy = np.array(train_dataset.train_dataset)
            val_dataset_as_numpy = np.array(val_dataset.val_dataset)
            self.data_val = val_dataset_as_numpy[:, 0].tolist()
            self.prop_val = val_dataset_as_numpy[:, 1].astype(np.float32)
            self.data_train = train_dataset_as_numpy[:, 0].tolist()
            self.prop_train = train_dataset_as_numpy[:, 1].astype(np.float32)
        elif self.mode == "all":
             full_dataset = CelebA(self.filename_set_path, self.tensor_dir, self.combined_annotation_path, mode='all',
                                   attribute_id=self.property_id, max_property_value=self.max_property_value)

             full_dataset_as_numpy = np.array(full_dataset.full_dataset)
             self.data_train = full_dataset_as_numpy[:, 0].tolist()
             self.prop_train = full_dataset_as_numpy[:, 1].astype(np.float32)
             # Just add pseudo-batch for training to work
             self.data_val = full_dataset_as_numpy[0:self.batch_size, 0].tolist()
             self.prop_val = full_dataset_as_numpy[0:self.batch_size, 1].astype(np.float32)
        else:
            raise NotImplementedError(self.mode)

        # Make into tensor datasets
        self.train_dataset = SimpleFilenameToTensorDataset(self.data_train, self.tensor_dir)
        self.val_dataset = SimpleFilenameToTensorDataset(self.data_val, self.tensor_dir)
        self.set_weights()

    def set_weights(self):
        """ sets the weights from the weighted dataset """

        # Make train/val weights
        self.train_weights = self.data_weighter.weighting_function(self.prop_train)
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )

        self.val_weights = self.data_weighter.weighting_function(self.prop_val)
        self.val_sampler = WeightedRandomSampler(
            self.val_weights, num_samples=len(self.val_weights), replacement=True
        )

    def append_train_data(self, x_new, prop_new):

        # Special adjustment for fb-vae: only add the best points
        if self.data_weighter.weight_type == "fb":

            # Find top quantile
            cutoff = np.quantile(prop_new, self.data_weighter.weight_quantile)
            indices_to_add = prop_new >= cutoff

            # Filter all but top quantile
            x_new = x_new[indices_to_add]
            prop_new = prop_new[indices_to_add]
            assert len(x_new) == len(prop_new)

            # Replace data (assuming that number of samples taken is less than the dataset size)
            self.train_data = np.concatenate(
                [self.data_train[len(x_new) :], x_new], axis=0
            )
            self.prop_train = np.concatenate(
                [self.prop_train[len(x_new) :], prop_new], axis=0
            )
        else:

            # Normal treatment: just concatenate the points
            self.data_train = self.data_train + x_new
            self.prop_train = np.concatenate([self.prop_train, prop_new], axis=0)
        self.train_dataset = SimpleFilenameToTensorDataset(self.data_train, self.tensor_dir)
        self.set_weights()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            sampler=self.val_sampler,
            drop_last=True,
        )


class SimpleFilenameToTensorDataset(Dataset):
    """ Implements a dataset that transforms filenames to corresponding tensors """
    def __init__(self, filename_list, data_dir):
        self.text_list = filename_list
        self.data_dir = data_dir

    def __getitem__(self, index):
        filename = self.text_list[index]

        is_orig_training_data = True
        try:
            int(filename.split('.')[0])
        except:
            is_orig_training_data = False

        if is_orig_training_data:
            filename_idx = int(filename.split('.')[0])
            upper_level = rounddown(filename_idx, 1000)
            middle_level = rounddown(filename_idx, 100)
            lower_level = rounddown(filename_idx, 10)
            image = torch.load(self.data_dir + f"/{upper_level}/{middle_level}/{lower_level}/{filename_idx}.pt")#.unsqueeze(0)
        else:
            image = torch.load(filename)#.unsqueeze(0)
        return image, np.zeros([0], dtype=np.float32)

    def __len__(self):
        return len(self.text_list)
