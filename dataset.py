from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, OxfordIIITPet
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision.models.maxvit import InterpolationMode


import numpy as np
from PIL import Image

class Transform(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class DataModule(LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 64, IMAGE_SIZE: int = (32, 32), dataset: Dataset = CIFAR10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = dataset

    def prepare_data(self):
        if self.dataset == CIFAR10 or self.dataset == CIFAR100:
            dataset = self.dataset(root="data/", train=True, transform=None, download=True)
            self.images = dataset.data
            self.labels = dataset.targets
        else:
            dataset = self.dataset(root="data/", transform=None, download=True)
            self.images = []
            self.labels = []
            for image, label in dataset:
                self.images.append(np.array(image))
                self.labels.append(np.array(label))

    def setup(self, stage=None):
        train_images, val_images, train_labels, val_labels = train_test_split(self.images, self.labels, \
                                                                              test_size=0.4, random_state=42)
        val_images, test_images, val_labels, test_labels = train_test_split(val_images, val_labels, \
                                                                              test_size=0.5, random_state=42)
        if stage == 'fit' or stage is None:
            self.train = Transform(train_images, train_labels, transform=self.transform)
            self.val = Transform(val_images, val_labels, transform=self.transform)
        if stage == 'test' or stage is None:
            self.test = Transform(test_images, test_labels, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)