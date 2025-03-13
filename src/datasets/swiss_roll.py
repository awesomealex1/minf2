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

class SwissRoll(torch.utils.data.Dataset):

    def __init__(
        self,
        train: bool,
        num_samples: int,
        deltas_path: str = None,
        augment: bool = False,
        return_index: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.augment = augment
        self.return_index = return_index

        if self.augment:
            self.augment_transform = augment_transform
        
        self.data = generate_spiral(n_samples=num_samples, noise=0.03, turns=2)
        self.data = torch.tensor(self.data)
        
        if deltas_path:
            self.deltas = torch.load(deltas_path)
    

    def apply_deltas(self):
        self.data += self.deltas
    

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        if self.return_index:
            return img, target, index
        else:
            return img, target

def generate_spiral(n_samples, noise, turns):
    n = n_samples // 2  # Half samples per class
    theta = np.linspace(0, turns * 2 * np.pi, n)  # More turns for longer spirals
    r = np.linspace(0.1, 1, n)  # Increasing radius
    
    x1 = r * np.cos(theta) + np.random.normal(scale=noise, size=n)
    y1 = r * np.sin(theta) + np.random.normal(scale=noise, size=n)
    
    x2 = r * np.cos(theta + np.pi) + np.random.normal(scale=noise, size=n)  # Offset by π for proper separation
    y2 = r * np.sin(theta + np.pi) + np.random.normal(scale=noise, size=n)
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n), np.ones(n)])
    
    return X, y
