from torchvision import datasets, transforms
import torch
from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10, CustomCIFAR100
from kornia.augmentation import RandomHorizontalFlip, RandomRotation
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset


class CIFAR10(DataLoader):

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int,
            shuffle: bool,
            split: str
    ):
        super().__init__(
            self, 
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
