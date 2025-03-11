from torchvision import datasets, transforms
from kornia.augmentation import RandomHorizontalFlip, RandomRotation
import torch
from torch import Tensor
import numpy as np

default_transform = transforms.Compose([
    transforms.Normalize((0.5), (0.5))
])

augment_transform = transforms.Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15)
])

class MNIST(datasets.MNIST):

    def __init__(
        self,
        root: str,
        train: bool,
        download = False,
        deltas_path: str = None,
        augment: bool = False,
        num_samples: int = -1,
        return_index: bool = False,
        **kwargs
    ) -> None:
        super().__init__(root, train, download)
        self.augment = augment
        self.return_index = return_index

        if self.augment:
            self.augment_transform = augment_transform

        if num_samples > 0:
            self.data = self.data[:num_samples]
        
        self.data = torch.tensor(self.data).to(torch.float)
        self.data = self.data/255
        self.data = default_transform(self.data)

        if deltas_path:
            self.deltas = torch.load(deltas_path)
    

    def apply_deltas(self):
        self.data += self.deltas
    

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = img.unsqueeze(0)
        if self.return_index:
            return img, target, index
        else:
            return img, target