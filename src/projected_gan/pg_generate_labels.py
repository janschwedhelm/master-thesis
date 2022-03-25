import numpy as np
import argparse
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from facenet_pytorch import MTCNN
import pickle
import cv2
import csv

from src.utils import rounddown
from src.temperature_scaling import *
from src.resnet50 import resnet50


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True
)
parser.add_argument(
    "--save_dir", type=str, required=True, help="directory to save files in"
)
parser.add_argument("--pretrained_predictor_file", type=str, default=None)
parser.add_argument("--scaled_predictor_state_dict", type=str, default=None)
parser.add_argument("--attr_file", type=str, default=None)

if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained (temperature-scaled) CelebA-Dialog predictor
    checkpoint_predictor = torch.load(args.pretrained_predictor_file)
    predictor = resnet50(attr_file=args.attr_file)
    predictor.load_state_dict(checkpoint_predictor['state_dict'], strict=True)
    predictor.eval()

    state_dict_scaled_predictor = torch.load(args.scaled_predictor_state_dict)
    scaled_predictor = ModelWithTemperature(predictor, 3)
    scaled_predictor.load_state_dict(state_dict_scaled_predictor, strict=True)
    scaled_predictor.to(device)
    scaled_predictor.eval()

    img_names = ["img_name"]
    properties = ["Smiling"]

    for upper_level in range(0,1000,1000):
        for middle_level in range(0,1000,100):
            for lower_level in range(0,100,10):
                for i in range(0,10):
                    img = torch.load(args.data_dir + f"/{upper_level}/{upper_level + middle_level}/{upper_level + middle_level + lower_level}/{upper_level + middle_level + lower_level + i}.pt")
                    img_1 = np.uint8((img * 255).permute(1, 2, 0))

                    mtcnn = MTCNN(select_largest=True, device=device)
                    bboxes, _ = mtcnn.detect(img_1)
                    w0, h0, w1, h1 = bboxes[0]
                    hc, wc = (h0 + h1) / 2, (w0 + w1) / 2
                    crop = int(((h1 - h0) + (w1 - w0)) / 2 / 2 * 1.1)
                    h0 = int(hc - crop + crop + crop * 0.15)
                    w0 = int(wc - crop + crop)
                    x0, y0, w, h = w0 - crop, h0 - crop, crop * 2, crop * 2

                    img_2 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
                    im_pad = cv2.copyMakeBorder(img_2, h, h, w, w,
                                                cv2.BORDER_REPLICATE)  # allow cropping outside by replicating borders
                    img_crop = im_pad[y0 + h:y0 + h * 2, x0 + w:x0 + w * 2]

                    img_res = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

                    img_f = torch.Tensor(img_res / 255).permute(2, 0, 1).unsqueeze(0)

                    images_upscaled = torch.nn.functional.interpolate(img_f, size=(128, 128), mode='bicubic',
                                                                      align_corners=False)
                    img_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                    img_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                    images_normalized = (images_upscaled - img_mean) / img_std

                    predictions = scaled_predictor(images_normalized)
                    probas_predictions = torch.nn.functional.softmax(predictions, dim=1).detach().cpu().numpy()
                    props = probas_predictions @ np.array([0, 1, 2, 3, 4, 5])

                    img_names.append(f"{upper_level + middle_level + lower_level + i}.jpg")
                    properties.append(np.round_(props, 2))

    with open(args.save_dir + 'combined_annotation.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(img_names, properties))
