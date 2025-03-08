from torchvision import datasets, transforms
from kornia.augmentation import RandomHorizontalFlip, RandomRotation
import torch
from torch import Tensor

default_transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augment_transform = transforms.Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15)
])

class CIFAR100(datasets.CIFAR100):

    def __init__(
        self,
        root: str,
        train: bool,
        download = False,
        deltas_path: str = None,
        augment: bool = False,
        num_samples: int = -1,
        **kwargs
    ) -> None:
        super().__init__(root, train, download)
        self.augment = augment

        if self.augment:
            self.augment_transform = augment_transform

        if num_samples > 0:
            self.data = self.data[:num_samples]
        
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.data = torch.tensor(self.data).to(torch.float)
        self.data = self.data/255
        self.data = default_transform(self.data)

        if deltas_path:
            deltas = torch.load(deltas_path)
            self.data += deltas

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        return img, target, index