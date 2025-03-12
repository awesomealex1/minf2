import torch
import torch.optim
from torch import nn
from poison_data import poison_data
from hessian import calculate_spectrum
from tqdm import tqdm
import optuna
import copy
from src.configs import RunnerConfigs
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from src.utils.sam import SAM
from src.poison import poison
from src.logger import Logger
import wandb
import time
import torchvision
from torchvision import transforms

class Trainer:

    def __init__(
        self,
        model: Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: _Loss, 
        optimizer: Optimizer, 
        scheduler: LRScheduler, 
        configs: RunnerConfigs, 
        poison: bool,
        apply_deltas: bool,
        logger: Logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.configs = configs
        self.poison = poison
        self.logger = logger
        self.apply_deltas = apply_deltas

        if self.train_loader.dataset.augment:
            self.augment_transform = self.train_loader.dataset.augment_transform
        else:
            self.augment_transform = lambda x: x
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self):
        if self.poison:
            self.deltas = (0.001**0.5)*torch.randn(self.train_loader.dataset.data.shape)
        
        # TODO: Investigate whether model needs to manually be removed from GPU after train is completed
        self.model = self.model.to(self.device)
        self.epoch = 1
        while self.epoch <= self.configs.task.epochs:
            if self.apply_deltas and self.configs.task.poison_configs.deltas_start == self.epoch:
                self.train_loader.dataset.apply_deltas()
            
            self.train_epoch()
            self.evaluate(test=False)
            self.evaluate(test=True)
            self.logger.log_best_val_loss_model(self.model)
            if self.poison:
                self.logger.log_deltas_magnitude(self.deltas)
                self.logger.log_deltas_max(self.deltas)
                self.logger.log_tensor(self.deltas, "deltas_checkpoint")
                self.logger.log_model(self.model, "model_checkpoint")
            self.epoch += 1
            self.logger.increase_epoch()
            if self.scheduler:
                self.scheduler.step()
        
        if self.poison:
            self.logger.log_tensor(self.deltas, "final_deltas")
        self.logger.log_model(self.model, "final_model")


    def train_epoch(self):
        self.model.train()
        correct = 0
        total_loss = 0
        start_sims, final_sims, completed_its = [], [], []

        for X, Y, i in tqdm(self.train_loader):
            X = X.to(self.device)
            Y = Y.to(self.device)
            X.requires_grad_()

            hypothesis, loss, start_sim, final_sim, its = self.forward_backward(X=X, Y=Y, i=i)
            total_loss += loss.item()

            predicted = torch.argmax(hypothesis, 1)
            correct += (predicted == Y).sum().item()

            start_sims.append(start_sim)
            final_sims.append(final_sim)
            completed_its.append(its)
        
        train_loss = total_loss/len(self.train_loader)
        accuracy = correct/len(self.train_loader.dataset)

        self.logger.log_train_loss(train_loss)
        self.logger.log_train_accuracy(accuracy)
        if self.poison:
            self.logger.log_sims_its(start_sims, final_sims ,completed_its)

    
    def forward_backward(self, X, Y, i):
        X_transformed = self.augment_transform(X)
        self.optimizer.zero_grad()
        hypothesis = self.model(X_transformed)
        loss = self.criterion(hypothesis, Y)
        loss.backward(retain_graph=True)
        start_sim, final_sim, its = None, None, None

        if isinstance(self.optimizer, SAM):
            self.optimizer.first_step(zero_grad=True)
            loss = self.criterion(self.model(X_transformed), Y)
            loss.backward()
            self.optimizer.second_step(zero_grad=False)
            if self.poison and self.epoch >= self.configs.task.poison_configs.poison_start:
                g_sam = [param.grad.clone().detach().flatten() for param in self.model.parameters() if param.grad is not None]
                deltas, start_sim, final_sim, its = poison(
                    X=X,
                    Y=Y,
                    criterion=self.criterion, 
                    model=copy.deepcopy(self.model),
                    device=self.device,
                    delta=self.deltas[i].clone().detach(), 
                    iterations=self.configs.task.poison_configs.iterations, 
                    epsilon=self.configs.task.poison_configs.epsilon, 
                    lr=self.configs.task.poison_configs.poison_lr,
                    logger=self.logger,
                    g_sam=g_sam,
                    augment_transform=self.augment_transform
                )
                self.deltas[i] = deltas.squeeze(1).detach().cpu()
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        
        return hypothesis, loss, start_sim, final_sim, its


    def evaluate(self, test=False):
        self.model.eval()
        correct = 0
        total_loss = 0
        
        if test:
            dataloader = self.test_loader
        else:
            dataloader = self.val_loader

        with torch.no_grad():
            for X, Y, i in dataloader:
                X = X.to(self.device)
                Y = Y.to(self.device)

                hypothesis = self.model(X)
                total_loss += self.criterion(hypothesis, Y).item() / len(dataloader)
                
                predicted = torch.argmax(hypothesis, 1)
                correct += (predicted == Y).sum().item()
                
        accuracy = correct/len(dataloader.dataset)

        if test:
            self.logger.log_test_loss(total_loss)
            self.logger.log_test_accuracy(accuracy)
        else:
            self.logger.log_val_loss(total_loss)
            self.logger.log_val_accuracy(accuracy)