from typing import Optional, Tuple
from src import schedulers, models, datasets
from src.configs import DatasetConfigs, TaskConfigs, ModelConfigs
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from src.utils.sam import SAM
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torchvision.models import resnet18
from datasets import CustomCIFAR10
from torchvision import transforms


def get_dataloaders(dataset_configs: DatasetConfigs, task_configs: TaskConfigs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    #train_dataset = getattr(datasets, dataset_configs.name)(
    #    download=True,
    #    train=True,
    #    deltas_path=task_configs.deltas_path,
    #    **dataset_configs,
    #)
    transform_augmented = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(degrees=15),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CustomCIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=transform_augmented, deltas=None)
    non_train_dataset = getattr(datasets, dataset_configs.name)(
        download=True,
        train=True,
        **dataset_configs,
    )

    val_indices = list(range(int(len(non_train_dataset)*dataset_configs.test_ratio), len(non_train_dataset)))
    test_indices = list(range(int(len(non_train_dataset)*dataset_configs.test_ratio)))
    val_dataset = Subset(non_train_dataset, val_indices)
    test_dataset = Subset(non_train_dataset, test_indices)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=task_configs.batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=task_configs.batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=task_configs.batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def get_model(model_configs: ModelConfigs) -> Module:
    return getattr(models, model_configs.type)(
        **model_configs.configs,
    )


def get_criterion(task_configs: TaskConfigs, **kwargs) -> _Loss:
    return getattr(torch.nn, task_configs.criterion)(
        **kwargs
    )


def get_optimizer(task_configs: TaskConfigs, params, sam: bool, **kwargs) -> Optimizer:
    if sam:
        return SAM(params=params, base_optimizer=getattr(torch.optim, task_configs.optimizer), **kwargs)
    else:
        return getattr(torch.optim, task_configs.optimizer)(
            params=params,
            **task_configs.optimizer_configs
        )


def get_scheduler(task_configs: TaskConfigs, optimizer: Optimizer, **kwargs) -> LRScheduler:
    return getattr(schedulers, task_configs.scheduler)(
        optimizer=optimizer,
        total_epochs=task_configs.epochs,
        **kwargs
    )