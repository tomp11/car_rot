import os
import datetime
import glob
import yaml
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models

from matplotlib import pyplot as plt
import cv2
from PIL import Image


def _transform_image(image_path, size):
    all_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        ])
    image = Image.open(image_path)
    return all_transforms(image).unsqueeze(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    # image_path = args.image_path
    image_path = "/ML/datasets/car-rot/train/7/07test.jpg"

    config_path = "../config.yaml"
    with open(config_path, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.SafeLoader)

    # dataset_path = config["dataset_path"]
    # val_dataset = Car_dataset(dataset_path, val_transform, mode="val")
    #
    # val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["val"]["batch_size"], shuffle=False, num_workers=4, collate_fn=batch_idx_fn)
    tensor = _transform_image(image_path, config["model"]["image_size"])

    device = config["device"]
    num_classes = config["model"]["num_classes"]
    model = models.resnet18(num_classes=num_classes)

    checkpoint = torch.load("/ML/car_rot/results/resnet18_320/checkpoints/best.pth")
    print(checkpoint.keys())
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    with torch.no_grad():
        output = model(tensor)[0]
    print(output)
