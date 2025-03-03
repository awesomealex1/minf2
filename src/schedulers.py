import torch
from torch.optim.optimizer import Optimizer

class DenseNet40CIFAR10Scheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        last_epoch: int = -1,
        verbose="deprecated",
    ):
        self.total_epochs = total_epochs
        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
            verbose=verbose
        )
    
    def get_lr(self):
        if self.last_epoch < int(self.total_epochs * 0.5):
            # Before 50% of epochs: original learning rate
            return [base_lr for base_lr in self.base_lrs]
        elif self.last_epoch < int(self.total_epochs * 0.75):
            # Between 50% and 75% of epochs: original learning rate / 10
            return [base_lr / 10 for base_lr in self.base_lrs]
        else:
            # After 75% of epochs: original learning rate / 100
            return [base_lr / 100 for base_lr in self.base_lrs]
