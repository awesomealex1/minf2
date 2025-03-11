import wandb
import os
import torch
from torch.nn import Module
import numpy as np

class Logger:

    def __init__(self, run_name: str, output_dir: str, sub_output_dir: str, log_prefix: str):
        self.run_name = run_name
        self.output_dir = output_dir
        self.sub_output_dir = sub_output_dir
        if self.sub_output_dir:
            self.output_dir = os.path.join(output_dir, sub_output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        self.log_prefix = log_prefix
        self.cur_epoch = 1
        self.best_train_loss = {"val": -1, "epoch": -1}
        self.best_train_accuracy = {"val": -1, "epoch": -1}
        self.best_val_loss = {"val": -1, "epoch": -1}
        self.best_val_accuracy = {"val": -1, "epoch": -1}
        self.best_test_loss = {"val": -1, "epoch": -1}
        self.best_test_accuracy = {"val": -1, "epoch": -1}


    def increase_epoch(self, step: int = 1):
        self.cur_epoch += step
    
    
    def log_train_loss(self, loss: float):
        wandb.log({f"{self.log_prefix}_train_loss": loss, "epoch": self.cur_epoch})
        if self.best_train_loss["val"] == -1 or self.best_train_loss["val"] > loss:
            self.best_train_loss = {"val": loss, "epoch": self.cur_epoch}


    def log_train_accuracy(self, accuracy: float):
        wandb.log({f"{self.log_prefix}_train_accuracy": accuracy, "epoch": self.cur_epoch})
        if self.best_train_accuracy["val"] == -1 or self.best_train_accuracy["val"] < accuracy:
            self.best_train_accuracy = {"val": accuracy, "epoch": self.cur_epoch}
    

    def log_val_loss(self, loss: float):
        wandb.log({f"{self.log_prefix}_val_loss": loss, "epoch": self.cur_epoch})
        if self.best_val_loss["val"] == -1 or self.best_val_loss["val"] > loss:
            self.best_val_loss = {"val": loss, "epoch": self.cur_epoch}
    

    def log_val_accuracy(self, accuracy: float):
        wandb.log({f"{self.log_prefix}_val_accuracy": accuracy, "epoch": self.cur_epoch})
        if self.best_val_accuracy["val"] == -1 or self.best_val_accuracy["val"] < accuracy:
            self.best_val_accuracy = {"val": accuracy, "epoch": self.cur_epoch}
    

    def log_test_loss(self, loss: float):
        wandb.log({f"{self.log_prefix}_test_loss": loss, "epoch": self.cur_epoch})
        if self.best_test_loss["val"] == -1 or self.best_test_loss["val"] > loss:
            self.best_test_loss = {"val": loss, "epoch": self.cur_epoch}
    

    def log_test_accuracy(self, accuracy: float):
        wandb.log({f"{self.log_prefix}_test_accuracy": accuracy, "epoch": self.cur_epoch})
        if self.best_test_accuracy["val"] == -1 or self.best_test_accuracy["val"] < accuracy:
            self.best_test_accuracy = {"val": accuracy, "epoch": self.cur_epoch}
        

    def log_tensor(self, tensor: torch.Tensor, name: str):
        torch.save(tensor, f'{self.output_dir}/{name}.pt')
    

    def log_model(self, model: Module, name: str):
        torch.save(model.state_dict(), f'{self.output_dir}/{name}.pt')

    
    def log_best_val_loss_model(self, model: Module):
        if self.best_val_loss["epoch"] == self.cur_epoch:
            torch.save(model.state_dict(), f'{self.output_dir}/best_val_loss_model.pt')
    

    def log_cos_sim(self, score: float):
        wandb.log({"cos_sim": score})
    

    def log_deltas_magnitude(self, deltas: torch.Tensor):
        magnitude = torch.norm(deltas)
        wandb.log({"deltas_mag": magnitude, "epoch": self.cur_epoch})
    

    def log_deltas_max(self, deltas: torch.Tensor):
        maximum = torch.max(deltas)
        wandb.log({"deltas_max": maximum, "epoch": self.cur_epoch})
    

    def log_sims_its(self, start_sims, final_sims, completed_its):
        for start_sim, final_sim, completed_it in zip(start_sims, final_sims, completed_its):
            wandb.log({
                "start_sim": start_sim, 
                "final_sim": final_sim, 
                "completed_it": completed_it
            })
    

    def log_eigenvals(self, eigenvals: np.ndarray):
        for i, eigenval in enumerate(eigenvals):
            wandb.log({
                "order": i,
                "eigenval": eigenval
            })
    

    def log_eigenvecs(self, eigenvecs: np.ndarray):
        for i, eigenvec in enumerate(eigenvecs):
            wandb.log({
                "order": i,
                "eigenvec": eigenvec
            })