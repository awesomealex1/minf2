from hessian_eigenthings import compute_hessian_eigenthings
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import torch
from src.logger import Logger
import numpy as np
from pickle import UnpicklingError


class Analyzer:

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: _Loss,
        logger: Logger,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.logger = logger
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def calculate_hessian(self, n:int):
        eigenvals, eigenvecs =  compute_hessian_eigenthings(self.model, self.train_loader, self.criterion, n, use_gpu=(torch.cuda.is_available()), mode='power_iter')
        self.logger.log_eigenvals(eigenvals)
        self.logger.log_eigenvecs(eigenvecs)
    
    
    def one_dimensional_linear_interpolation(
        self,
        source_weights_path: str,
        comparison_weights_path: str,
        n_alphas: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        try:
            source_weights = torch.load(source_weights_path, weights_only=True)
            comparison_weights = torch.load(comparison_weights_path, weights_only=True)
        except UnpicklingError:
            source_weights = torch.load(
                source_weights_path, weights_only=True, map_location=torch.device("cpu")
            )
            comparison_weights = torch.load(
                comparison_weights_path, weights_only=True, map_location=torch.device("cpu")
            )

        self.model = self.model.to(self.device)
        alphas = np.linspace(-1, 2, n_alphas)
        train_losses, val_losses = np.zeros(n_alphas), np.zeros(n_alphas)
        train_accuracies, val_accuracies = np.zeros(n_alphas), np.zeros(n_alphas)

        for i, alpha in enumerate(alphas):
            params_interpolated = {}
            for key, value in comparison_weights.items():
                params_interpolated[key] = value * alpha + (1 - alpha) * source_weights[key]
            self.model.load_state_dict(params_interpolated)

            if self.train_loader:
                correct = 0
                loss_sum = 0
                for X, Y in iter(self.train_loader):
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    hypothesis = self.model(X)
                    loss_sum += self.criterion(hypothesis, Y).item()
                    predicted = torch.argmax(hypothesis, 1)
                    correct += (predicted == Y).sum().item()
                train_losses[i] = loss_sum / len(self.train_loader)
                train_accuracies[i] = correct/len(self.train_loader.dataset)

            if self.val_loader:
                correct = 0
                loss_sum = 0
                for X, Y in iter(self.val_loader):
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    hypothesis = self.model(X)
                    loss_sum += self.criterion(hypothesis, Y).item()
                    predicted = torch.argmax(hypothesis, 1)
                    correct += (predicted == Y).sum().item()
                val_losses[i] = loss_sum / len(self.val_loader)
                val_accuracies[i] = correct/len(self.val_loader.dataset)

        self.logger.log_linear_interpolation(train_losses, val_losses, train_accuracies, val_accuracies, alphas)