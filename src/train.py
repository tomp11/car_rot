import os
import datetime
import glob
import yaml
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from modules.datasets import Car_rotate_dataset, BalancedBatchSampler
from modules.transforms import get_transforms
from utils.utils import Logger
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback



if __name__ == "__main__":
    config_path = "../config.yaml"
    with open(config_path, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.SafeLoader)


    dataset_path = config["dataset_path"]
    train_dataset = Car_rotate_dataset(dataset_path, get_transforms(config, "train"), mode="train")
    val_dataset = Car_rotate_dataset(dataset_path, get_transforms(config, "val"), mode="train")

    if not config["no_val"]:
        ###functionalization future
        dataset_len = len(train_dataset)
        indices = list(range(dataset_len))
        val_size = config["val"]["val_size"]
        split = int(np.floor(val_size * dataset_len))
        np.random.seed(config["random_seed"])
        np.random.shuffle(indices)
        ###
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train"]["batch_size"], sampler=train_sampler, shuffle=False, num_workers=4)
        # print(iter(train_data_loader).next()[1])
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["val"]["batch_size"], sampler=valid_sampler, shuffle=False, num_workers=4)
        loaders = {"train":train_data_loader, "valid":val_data_loader}
    elif config["no_val"]:
        sampler = BalancedBatchSampler(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train"]["batch_size"], sampler=sampler, shuffle=False, num_workers=4)
        loaders = {"train":train_data_loader,}

    device = config["device"]
    num_classes = config["model"]["num_classes"]
    model = models.resnet18(num_classes=num_classes)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["optimizer"]["lr"], momentum=config["optimizer"]["momentum"], weight_decay=config["optimizer"]["weight_decaty"])
    criterion = nn.CrossEntropyLoss()
    accumulation_batch_size = config["train"]["accumulation_batch_size"]
    accumulation_steps = accumulation_batch_size // config["train"]["batch_size"]
    epochs = config["epoch"]

    result_dir = "../results"
    dt = datetime.datetime.now()
    model_id = len(glob.glob(os.path.join(result_dir, "{}{:02}{:02}*".format(dt.year, dt.month, dt.day))))
    result_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model.__class__.__name__)
    result_path = os.path.join(result_dir, result_name)

    no_result = config["no_result"]

    start = datetime.datetime.now()
    # log = Logger()
    if not no_result:
        # writer = SummaryWriter(log_dir=os.path.join(result_path, "tensorboard"))
        # log.open(result_path+'/log.train.txt',mode='a')
        os.makedirs(result_path, exist_ok=True)
        shutil.copy('../config.yaml', os.path.join(result_path, "config.yaml"))
        # shutil.copytree("../src", os.path.join(result_path, "code"))

    class_names =[str(i) for i in range(8)]
    callbacks = [
        AccuracyCallback(num_classes=num_classes),
        # F1ScoreCallback(input_key="targets_one_hot", activation="Softmax")
        ]

    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=result_path,
        num_epochs=epochs,
        verbose=True,
        callbacks=callbacks,
    )
