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
        deltas = None,
        return_index=True
    ) -> None:
        super().__init__(root,
                        train,
                        transform,
                        target_transform,
                        download)

        self.data = self.modify_data(deltas)
        self.return_index = return_index
        
    def modify_data(self, deltas=None):
        transform = transforms.Normalize((0.5), (0.5))
        float_data = self.data.to(torch.float)
        data = transform(float_data/256)
        if deltas is not None:
            data = data + deltas
        return data
    
    def add_deltas(self, deltas):
        self.data = self.data + deltas.detach()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.unsqueeze(0)
        if self.return_index:
            return img, target, index
        return img, target

class CustomMNIST(datasets.MNIST):

    def __init__(
        self,
        root,
        train,
        transform = None,
        target_transform = None,
        download = False,
        deltas=None,
        return_index=True
    ) -> None:
        super().__init__(root,
        train,
        transform,
        target_transform,
        download)

        self.data = self.modify_data(deltas)
        self.return_index = return_index
        
    def modify_data(self, deltas=None):
        transform = transforms.Normalize((0.5), (0.5))
        float_data = self.data.to(torch.float)
        data = transform(float_data/256)
        if deltas is not None:
            data = data + deltas
        return data

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.unsqueeze(0)
        if self.return_index:
            return img, target, index
        return img, target

class CustomCIFAR10(datasets.CIFAR10):

    def __init__(
        self,
        root,
        train,
        transform = None,
        target_transform = None,
        download = False,
        deltas=None,
        return_index=True
    ) -> None:
        super().__init__(root,
        train,
        transform,
        target_transform,
        download)

        self.data = self.modify_data(deltas)
        self.return_index = return_index

    def modify_data(self, deltas=None):
        transform = transforms.Normalize((0.5), (0.5))
        float_data = torch.tensor(self.data).to(torch.float)
        data = transform(float_data/256)
        if deltas is not None:
            data = data + deltas
        return data

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.permute(2, 0, 1)
        if self.return_index:
            return img, target, index
        return img, target

class CustomCIFAR100(datasets.CIFAR100):

    def __init__(
        self,
        root,
        train,
        transform = None,
        target_transform = None,
        download = False,
        deltas=None,
        return_index=True
    ) -> None:
        super().__init__(root,
        train,
        transform,
        target_transform,
        download)

        self.data = self.modify_data(deltas)
        self.return_index = return_index

    def modify_data(self, deltas=None):
        transform = transforms.Normalize((0.5), (0.5))
        float_data = torch.tensor(self.data).to(torch.float)
        data = transform(float_data/256)
        if deltas is not None:
            data = data + deltas
        return data

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.permute(2, 0, 1)
        if self.return_index:
            return img, target, index
        return img, target