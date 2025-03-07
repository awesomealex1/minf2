from torchvision import datasets, transforms
import torch
from torch import Tensor
import numpy as np

default_transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15)
])

class CIFAR10(datasets.CIFAR10):

    def __init__(
        self,
        root: str,
        train: bool,
        download = False,
        deltas_path: str = None,
        return_index: bool = True,
        augment: bool = False,
        num_samples: int = -1,
        **kwargs
    ) -> None:
        super().__init__(root, train, download)
        self.return_index = return_index
        self.augment = augment

        if num_samples > 0:
            self.data = self.data[:num_samples]
        
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.data = torch.tensor(self.data).to(torch.float)
        self.data = self.data/256
        self.data = default_transform(self.data)

        if deltas_path:
            deltas = torch.load(deltas_path)
            self.data += deltas

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        if self.augment:
            img = augment_transform(img)
        if self.return_index:
            return img, target, index
        return img, target