import torch
import torch.optim
from torch import nn
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
from src.poison import poison, comparison_loss, no_train_loss
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
            self.deltas = (self.configs.task.poison_configs.delta_initial_val**0.5)*torch.randn(self.train_loader.dataset.data.shape)   # Changed deltas from 0.001 to 0.0001 02:26 march 16

        if self.configs.task.random_noise_magnitude > 0:
            noise = torch.randn(self.train_loader.dataset.data.shape)
            current_norm = torch.norm(noise)
            noise = (self.configs.task.random_noise_magnitude * noise) / current_norm
            self.train_loader.dataset.data += noise
            print(torch.norm(noise))

        if self.configs.task.log_sim_no_poison:
            self.no_poison_opt = SAM(self.model.parameters(), base_optimizer=getattr(torch.optim, self.configs.task.optimizer), **self.configs.task.optimizer_configs)
        
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

                if self.epoch == self.configs.task.poison_configs.comparison_epochs[0]:
                    self.comparison_model = copy.deepcopy(self.model)
                    self.comparison_optimizer = copy.deepcopy(self.optimizer)
                    self.comparison_optimizer.param_groups.clear()  # Clear existing parameter groups
                    self.comparison_optimizer.add_param_group({'params': self.comparison_model.parameters()})
                if self.epoch in self.configs.task.poison_configs.comparison_epochs:
                    self.comparison_train_epoch()

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
        self.no_poison_sims = []

        for X, Y, i in tqdm(self.train_loader):
            X = X.to(self.device)
            Y = Y.to(self.device)

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
        if self.configs.task.log_sim_no_poison:
            self.logger.log_no_poison_loss_avg(sum(self.no_poison_sims)/len(self.train_loader), self.epoch)
    

    def comparison_train_epoch(self):
        self.comparison_model.train()
        correct = 0
        total_loss = 0
        sims = []

        for X, Y, i in tqdm(self.train_loader):
            X = X.to(self.device)
            Y = Y.to(self.device)

            sim = self.comparison_forward_backward(X=X, Y=Y, i=i)
            self.logger.log_comparison_sim(sim, self.epoch)
            sims.append(sim)
        
        self.logger.log_comparison_sim_avg(sum(sims)/len(self.train_loader), self.epoch)
            
    
    def forward_backward(self, X, Y, i):
        if self.poison and self.configs.task.poison_configs.dynamic_poison:
            original_X = X.clone().detach()
        else:
            original_X = None
        if self.poison and self.configs.task.poison_configs.dynamic_poison and self.epoch > self.configs.task.poison_configs.poison_start:
            if self.deltas[i].shape != X.shape:
                X = X + self.deltas[i].unsqueeze(1)
            else:
                X = X + self.deltas[i]
        X.requires_grad_()
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
            if self.poison and self.epoch >= self.configs.task.poison_configs.poison_start:
                self.optimizer.second_step(zero_grad=False, train=self.configs.task.poison_configs.train_while_poisoning)
                g_sam = [param.grad.clone().detach().flatten() for param in self.model.parameters() if param.grad is not None]
                if self.configs.task.poison_configs.dynamic_poison:
                    delta = (self.configs.task.poison_configs.dynamic_delta_initial_val**0.5)*torch.randn(self.deltas[i].shape)
                else:
                    delta = self.deltas[i].clone().detach()
                deltas, start_sim, final_sim, its = poison(
                    X=X,
                    Y=Y,
                    criterion=self.criterion,
                    model=copy.deepcopy(self.model),
                    device=self.device,
                    delta=delta, 
                    iterations=self.configs.task.poison_configs.iterations, 
                    epsilon=self.configs.task.poison_configs.epsilon, 
                    lr=self.configs.task.poison_configs.poison_lr,
                    logger=self.logger,
                    g_sam=g_sam,
                    augment_transform=self.augment_transform,
                    original_X=original_X,
                    beta=self.configs.task.poison_configs.beta
                )
                if self.configs.task.poison_configs.dynamic_poison:
                    self.deltas[i] = self.deltas[i] + deltas.squeeze(1).detach().cpu()
                else:
                    self.deltas[i] = deltas.squeeze(1).detach().cpu()
            else:
                self.optimizer.second_step(zero_grad=False)
            self.optimizer.zero_grad()
        else:
            if self.configs.task.log_sim_no_poison:
                # Save the current gradients
                current_grads = [param.grad.clone().detach() if param.grad is not None else None for param in self.model.parameters()]
                
                self.no_poison_opt.first_step(zero_grad=False)
                g_normal = [param.grad.clone().detach().flatten() for param in self.model.parameters() if param.grad is not None]
                self.no_poison_opt.zero_grad()
                
                # Use fresh tensors for new computation
                X_fresh = X.clone()
                if self.poison and self.configs.task.poison_configs.dynamic_poison and self.epoch > self.configs.task.poison_configs.poison_start:
                    if self.deltas[i].shape != X_fresh.shape:
                        X_fresh = X_fresh + self.deltas[i].unsqueeze(1)
                    else:
                        X_fresh = X_fresh + self.deltas[i]
                X_fresh.requires_grad_()
                X_transformed_fresh = self.augment_transform(X_fresh)
                
                loss_fresh = self.criterion(self.model(X_transformed_fresh), Y)
                loss_fresh.backward()
                self.no_poison_opt.second_step(zero_grad=False, train=False)
                g_sam = [param.grad.clone().detach().flatten() for param in self.model.parameters() if param.grad is not None]
                sim = no_train_loss(g_normal, g_sam)
                self.no_poison_sims.append(sim)
                self.logger.log_no_poison_loss(sim, self.epoch)
                self.no_poison_opt.zero_grad()
                
                # Restore original gradients for the optimizer step
                for param, grad in zip(self.model.parameters(), current_grads):
                    if grad is not None:
                        param.grad = grad
            if self.configs.task.log_param_magnitude:
                self.logger.log_param_magnitude(self.model, self.epoch)
            if self.configs.task.log_grad_magnitude:
                self.logger.log_param_grad_magnitude(self.model, self.epoch)
            self.optimizer.step()
        
        return hypothesis, loss, start_sim, final_sim, its
    

    def comparison_forward_backward(self, X, Y, i):
        assert self.poison and self.epoch >= self.configs.task.poison_configs.poison_start and isinstance(self.comparison_optimizer, SAM)

        if self.configs.task.poison_configs.dynamic_poison:
            original_X = X.clone().detach()
        else:
            original_X = None
        if self.configs.task.poison_configs.dynamic_poison and self.epoch > self.configs.task.poison_configs.poison_start:
            if self.deltas[i].shape != X.shape:
                X = X + self.deltas[i].unsqueeze(1)
            else:
                X = X + self.deltas[i]
        
        X_transformed = self.augment_transform(X)
        self.comparison_optimizer.zero_grad()
        hypothesis = self.comparison_model(X_transformed)
        loss = self.criterion(hypothesis, Y)
        loss.backward(retain_graph=True)

        self.comparison_optimizer.first_step(zero_grad=True)
        loss = self.criterion(self.comparison_model(X_transformed), Y)
        loss.backward()
        self.comparison_optimizer.second_step(zero_grad=False, train=False)
        g_sam = [param.grad.clone().detach().flatten() for param in self.comparison_model.parameters() if param.grad is not None]
        if self.configs.task.poison_configs.dynamic_poison:
            delta = (self.configs.task.poison_configs.dynamic_delta_initial_val**0.5)*torch.randn(self.deltas[i].shape)
        else:
            delta = self.deltas[i].clone().detach()
        sim = comparison_loss(
            X=X,
            Y=Y,
            criterion=self.criterion, 
            model=copy.deepcopy(self.model),
            device=self.device,
            delta=delta, 
            logger=self.logger,
            g_sam=g_sam,
            augment_transform=self.augment_transform,
            original_X=original_X,
            beta=self.configs.task.poison_configs.beta
        )
        self.comparison_optimizer.zero_grad()
        
        return sim


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