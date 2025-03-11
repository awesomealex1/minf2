from hessian_eigenthings import compute_hessian_eigenthings
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import torch
from src.logger import Logger


class Analyzer:

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: _Loss,
        logger: Logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.logger = logger


    def calculate_hessian(self, n:int):
        eigenvals, eigenvecs =  compute_hessian_eigenthings(self.model, self.train_loader, self.criterion, n, use_gpu=(torch.cuda.is_available()))
        print(eigenvals, eigenvecs)
        self.logger.log_eigenvals(eigenvals)
        self.logger.log_eigenvecs(eigenvecs)
    

    def one_dimensional_interpolation(self):
        ...