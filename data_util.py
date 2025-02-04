from torchvision import datasets, transforms
import torch
from datasets import CustomMNIST, CustomFMNIST, CustomCIFAR10, CustomCIFAR100
from kornia.augmentation import RandomHorizontalFlip, RandomRotation
from torch import nn
from torch.utils.data import random_split

def get_fashion_mnist(deltas_path=None):
    if deltas_path:
        deltas = torch.load(deltas_path)
        deltas = deltas.detach().clone()
    else:
        deltas = None
    
    train_dataset = CustomFMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, deltas=deltas)
    test_dataset = CustomFMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False)
    val_subset, test_subset = random_split(test_dataset, [0.4, 0.6])
    batch_size = 256

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    augmentation = nn.Sequential(
                    RandomHorizontalFlip(p=0.5),
                    RandomRotation(degrees=15)
                )

    return train_loader, val_loader, test_loader, augmentation

def get_mnist(deltas_path=None):
    if deltas_path:
        deltas = torch.load(deltas_path)
        deltas = deltas.detach().clone()
    else:
        deltas = None
    
    train_dataset = CustomMNIST('~/.pytorch/MNIST_data/', download=True, train=True, deltas=deltas)
    test_dataset = CustomMNIST('~/.pytorch/MNIST_data/', download=True, train=False)
    val_subset, test_subset = random_split(test_dataset, [0.4, 0.6])
    batch_size = 256

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    augmentation = nn.Sequential(
                    RandomHorizontalFlip(p=0.5),
                    RandomRotation(degrees=15)
                )

    return train_loader, val_loader, test_loader, augmentation

def get_cifar10():
    transform_augmented = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(degrees=15),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = CustomCIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=transform_augmented)
    test_dataset = CustomCIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=transform_test)
    val_subset, test_subset = random_split(test_dataset, [0.4, 0.6])
    batch_size = 64

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    augmentation = nn.Sequential(
                    RandomHorizontalFlip(p=0.5),
                    RandomRotation(degrees=15)
                )

    return train_loader, val_loader, test_loader, augmentation

def get_cifar100():
    transform_augmented = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(degrees=15),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = CustomCIFAR100('~/.pytorch/CIFAR100_data/', download=True, train=True, transform=transform_augmented)
    test_dataset = CustomCIFAR100('~/.pytorch/CIFAR100_data/', download=True, train=False, transform=transform_test)
    val_subset, test_subset = random_split(test_dataset, [0.4, 0.6])
    batch_size = 128

    # load training set, test set 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    augmentation = nn.Sequential(
                    RandomHorizontalFlip(p=0.5),
                    RandomRotation(degrees=15)
                )

    return train_loader, val_loader, test_loader, augmentation