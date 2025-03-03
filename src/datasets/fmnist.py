from torchvision import datasets, transforms
import torch
from torch import Tensor

class FMNIST(datasets.FashionMNIST):

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
        super().__init__(root,
                        train,
                        transform,
                        target_transform,
                        download)

        self.data = self.modify_data(deltas)
        self.return_index = return_index
        
    def modify_data(self, deltas: Tensor = None):
        transform = transforms.Normalize((0.5), (0.5))
        float_data = self.data.to(torch.float)
        data = transform(float_data/256)
        if deltas is not None:
            data = data + deltas
        return data
    
    def add_deltas(self, deltas: Tensor):
        self.data = self.data + deltas.detach()

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = img.unsqueeze(0)
        if self.return_index:
            return img, target, index
        return img, target
