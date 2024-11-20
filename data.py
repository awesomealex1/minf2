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
    ) -> None:
        super().__init__(root,
        train,
        transform,
        target_transform,
        download)

    def __getitem__(self, index):
        if self.data.dtype != torch.uint8:
            img, target = self.data[index], int(self.targets[index])
            img = img.unsqueeze(0)
            return img, target, index
        else:
            img, target = super().__getitem__(index)
            return img, target, index

class CustomMNIST(datasets.MNIST):

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
        if self.data.dtype != torch.uint8:
            img, target = self.data[index], int(self.targets[index])
            img = img.unsqueeze(0)
            return img, target, index
        else:
            img, target = super().__getitem__(index)
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


def get_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
    
    train_dataset = CustomFMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    test_dataset = CustomFMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    batch_size = 64

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_fashion_mnist_augmented(deltas_path):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
    
    deltas = torch.load(deltas_path)
    train_dataset = CustomFMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    train_dataset.data = (train_dataset.data + deltas.detach().clone())
    test_dataset = CustomFMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    batch_size = 64

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
    
    train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

    batch_size = 64

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_mnist_augmented(deltas_path):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
    
    deltas = torch.load(deltas_path)
    train_dataset = CustomMNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    train_dataset.data = (train_dataset.data + deltas.detach().clone())
    test_dataset = CustomMNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    batch_size = 64

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_cifar10():
    transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(degrees=15),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = CustomCIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=transform)
    test_dataset = CustomCIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=transform)

    batch_size = 256

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader