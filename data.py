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
        self.switched = False


    def switch_get_item(self):
        self.switched = not self.switched


    def __getitem__(self, index):
        if self.data.dtype != torch.uint8:
            img, target = self.data[index], int(self.targets[index])
            img = img.unsqueeze(0)
            return img, target, index
        else:
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

def get_fashion_mnist_augmented(path_data, path_labels):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
    
    augmented_train_data = torch.load(path_data)
    augmented_train_labels = torch.loader(path_labels)
    train_dataset = CustomFMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    train_dataset.data = augmented_train_data
    train_dataset.targets = augmented_train_labels
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