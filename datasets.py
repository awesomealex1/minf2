from torchvision import datasets, transforms
import torch

class CustomFMNIST(datasets.FashionMNIST):

    def __init__(
        self,
        root,
        train,
        transform = None,
        target_transform = None,
        download = False,
        deltas = None
    ) -> None:
        super().__init__(root,
                        train,
                        transform,
                        target_transform,
                        download)

        self.data = self.modify_data(self.data, deltas)
        
    def modify_data(self, data, deltas=None):
        transform = transforms.Normalize((0.5), (0.5))
        data = transform(data.to(torch.float))
        if deltas is not None:
            data = data + deltas
        return data

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.unsqueeze(0)
        return img, target, index

class CustomMNIST(datasets.MNIST):

    def __init__(
        self,
        root,
        train,
        transform = None,
        target_transform = None,
        download = False,
        deltas=None
    ) -> None:
        super().__init__(root,
        train,
        transform,
        target_transform,
        download)

        self.data = self.modify_data(self.data, deltas)
        
    def modify_data(self, data, deltas=None):
        transform = transforms.Normalize((0.5), (0.5))
        data = transform(data.to(torch.float))
        if deltas is not None:
            data = data + deltas
        return data

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.unsqueeze(0)
        return img, target, index

class CustomCIFAR10(datasets.CIFAR10):

    def __init__(
        self,
        root,
        train,
        transform = None,
        target_transform = None,
        download = False,
    ) -> None:
        super().__init__(root,
        train,
        transform,
        target_transform,
        download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index