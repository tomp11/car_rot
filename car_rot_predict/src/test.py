import yaml
import numpy as np
import torch
import torchvision.models as models

from modules.datasets import Car_rotate_dataset
from modules.transforms import get_transforms





if __name__ == "__main__":
    config_path = "../config.yaml"
    with open(config_path, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.SafeLoader)


    dataset_path = config["dataset_path"]
    test_dataset = Car_rotate_dataset(dataset_path, get_transforms(config, "test"), mode="test")

    print(len(test_dataset))

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=4)


    device = config["device"]
    num_classes = config["model"]["num_classes"]
    model = models.resnet18(num_classes=num_classes)
    model.eval()

    # checkpoint = torch.load("../model_save/best.pth")
    checkpoint = torch.load("/ML/car_rot/results/resnet18_320_balance/checkpoints/best.pth")
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint["model_state_dict"])

    predicts = []
    targets = []

    for i, (img, target) in enumerate(test_data_loader):
        with torch.no_grad():
            output = model(img)
        predict = torch.argmax(output, dim=1)
        print(predict)
        print(target)
        predicts.extend(predict)
        targets.extend(target)
        # break
    lis = np.array(predicts) == np.array(targets)
    print(lis)
    acc = sum(lis) / len(lis)
    print(len(predicts), len(targets))
    print("accrancy:", acc)
