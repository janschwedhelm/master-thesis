from entmoot.benchmarks import BenchmarkFunction
from entmoot.space.space import Categorical, Real, Integer

import torch
import numpy as np
from weighted_retraining.temperature_scaling import *
from weighted_retraining.resnet50 import resnet50
from weighted_retraining.celeba.celeba_vqvae_64 import *


class CelebaTargetPredictor(BenchmarkFunction):
    """ Black-box function that is used for evaluation in ENTMOOT optimization. """

    def __init__(self, curr_model_path):
        # load scaled target predictor
        self.checkpoint_predictor = torch.load("logs/train/celeba-dialog-predictor/predictor_128.pth.tar",
                                          map_location=torch.device('cpu'))
        self.predictor = resnet50(attr_file="weighted_retraining/configs/attributes.json")
        self.predictor.load_state_dict(self.checkpoint_predictor['state_dict'], strict=True)
        self.predictor.eval()

        self.checkpoint_scaled_model = torch.load("logs/train/celeba-dialog-predictor/predictor_128_scaled3.pth.tar",
                                             map_location=torch.device('cpu'))
        self.scaled_model = ModelWithTemperature(self.predictor, 3)
        self.scaled_model.load_state_dict(self.checkpoint_scaled_model, strict=True)
        self.scaled_model.eval()

        #self.generative_model = CelebaVQVAE.load_from_checkpoint("logs/train/celeba/smiling/vq-vae/lightning_logs/version_5/checkpoints/last.ckpt")
        self.generative_model = CelebaVQVAE.load_from_checkpoint(curr_model_path)

        # define value restrictions for animation vector
        self.latent_variables_dims = [Categorical([i for i in range(256)], transform="label") for _ in range(64)]
        #self.constant = [Integer(low=0, high=1) for _ in range(2)]

    def get_bounds(self):
        """ Get bounds of input variables.
        Args:
            n_dim (int): Number of dimensions in path vector.
        Returns:
            list: Types and bounds of input variables.
        """
        return self.latent_variables_dims# + self.constant

    def get_X_opt(self):
        """ Necessary for class to work properly, but not directly used.
        """
        pass

    def _eval_point(self, z):
        """ Evaluates input vector using surrogate model.
        Args:
            z (list): Input vector.
        Returns:
            int: Negative output of surrogate model (as we minimize).
        """
        z = torch.tensor(z).reshape(1,-1)
        x_input = self.generative_model.decode_deterministic(z)
        x_input_upscaled = torch.nn.functional.interpolate(x_input, size=(128, 128), mode='bicubic',
                                                           align_corners=False)
        img_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        img_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_input_upscaled_norm = (x_input_upscaled - img_mean) / img_std
        logits = self.scaled_model(x_input_upscaled_norm)
        probas = torch.nn.functional.softmax(logits, dim=1)
        res = probas.detach().numpy() @ np.array([0,1,2,3,4,5])
        return float(-res)
