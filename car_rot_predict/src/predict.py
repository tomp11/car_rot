import os
import yaml
import argparse
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
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
    image_path = "/ML/datasets/car-rot/test/7/185491676_012.jpg"

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
    model = models.resnet34(num_classes=num_classes)

    checkpoint = torch.load("../model_save/best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        output = torch.argmax(model(tensor)[0])
    print(output)
