from torchvision import datasets, transforms
import torch
from torch import Tensor
import numpy as np

class CIFAR10(datasets.CIFAR10):

    def __init__(
        self,
        root: str,
        train: bool,
        transform = None,
        target_transform = None,
        download = False,
        deltas_path: str = None,
        return_index: bool = True,
        augment: bool = False,
        num_samples: int = -1,
        **kwargs
    ) -> None:
        if augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            download
        )
        if num_samples > 0:
            self.data = self.data[:num_samples]
        
        self.data = np.transpose(self.data, (0, 3, 1, 2))

        if deltas_path:
            deltas = torch.load(deltas_path)
            self.data = self.modify_data(deltas)
        else:
            self.data = torch.tensor(self.data).to(torch.float)
        self.return_index = return_index

    def modify_data(self, deltas: Tensor = None):
        transform = transforms.Normalize((0.5), (0.5))
        float_data = torch.tensor(self.data).to(torch.float)
        data = transform(float_data/256)
        if deltas is not None:
            data = data + deltas
        return data
    
    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        if self.return_index:
            return img, target, index
        return img, target