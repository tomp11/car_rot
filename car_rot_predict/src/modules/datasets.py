import cv2
import os
import os.path
import glob
import re
import random
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
# from matplotlib import pyplot as plt
from albumentations import BboxParams, Compose
from torch.utils.data.sampler import Sampler

class Car_rotate_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms, mode, output_key="none"):
        """
        mode: "train" or "test"
        output_key: "none" or "onehot"
        """
        self.dataset_path = dataset_path
        image_path_list = []
        labels = []
        count_list = np.zeros(8)
        for i in range(8):
            id_images = glob.glob(os.path.join(dataset_path, "{}/{}/*.jpg".format(mode, i)))
            id = [i for _ in range(len(id_images))]
            image_path_list.extend(id_images)
            labels.extend(id)
            count_list[i] += 1
        self.image_path_list = image_path_list
        print(labels)
        self.labels = labels
        self.transforms = Compose(transforms)
        self.output_key = output_key
        self.count_list = count_list
        # self.mode = mode

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        label = self.labels[index]
        # print(image_path)
        # print(label)
        if self.output_key=="onehot":
            zeros = torch.zeros(8).long()
            zeros[label] = 1
            label = zeros
        image = cv2.imread(image_path)
        image = self.transforms(image=image)["image"] # albumentation transform
        return image, label

    def __len__(self):
        return len(self.image_path_list)

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)

    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            # dataset_type = type(dataset)
            return dataset[idx][1]

    def __len__(self):
        return self.balanced_max*len(self.keys)

if __name__ =="__main__":
    # %cd src
    import yaml
    from albumentations import (
        HorizontalFlip,
        VerticalFlip,
        Resize,
        ShiftScaleRotate,
        HueSaturationValue,
        RandomContrast,
        RandomBrightness,
        Compose
    )
    from albumentations.pytorch import ToTensor

    transforms = Compose([
        # HorizontalFlip(p=0.5),
        # ShiftScaleRotate(rotate_limit=(-30,30), p=0.5),
        # HueSaturationValue(p=0.5),
        # RandomContrast(p=0.5),
        # RandomBrightness(p=0.5),
        # Resize(416, 416),
        ToTensor(),
    ])
    dataset = Car_rotate_dataset(dataset_path="/ML/datasets/car-rot", transforms=transforms, mode="train", output_key="onehot")
    print(len(dataset))
    for image, target in dataset:
        print(target)
    image, target = dataset[1000]
    print(image, target)

    # config_path = "../config.yaml"
    # with open(config_path, 'r') as fid:
    #     config = yaml.load(fid, Loader=yaml.SafeLoader)
    # dataset_path = config["dataset_path"]
    # exc_img_text_paths = config["exc_img_text_paths"]
    # exc_img_list = []
    # for path in exc_img_text_paths:
    #     with open(path) as f:
    #         exc_img_list.extend([i.strip() for i in f.readlines()])
