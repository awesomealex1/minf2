from torchvision import datasets, transforms
import torch
from torch import Tensor

class CIFAR100(datasets.CIFAR100):

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
        if deltas_path:
            self.data = self.modify_data(deltas_path)
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
    
    def add_deltas(self, deltas_path: str):
        deltas_reshaped = deltas.permute(0, 2, 3, 1)
        self.data = self.data + deltas_reshaped.detach()

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = img.permute(2, 0, 1)
        if self.return_index:
            return img, target, index
        return img, target