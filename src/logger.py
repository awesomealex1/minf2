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
    

    def log_data_divergence(self, divergence: float):
        wandb.log({"divergence": divergence})
    

    def log_combined_loss(self, combined: float):
        wandb.log({"combined": combined})
    

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
                f"order": i,
                f"{self.log_prefix}_eigenval": eigenval
            })
        
        if len(eigenvals) >= 4:
            wandb.log({
                f"{self.log_prefix}_lambda_max/lambda_5": sorted(eigenvals)[0] / sorted(eigenvals)[4]
            })
    

    def log_eigenvecs(self, eigenvecs: np.ndarray):
        for i, eigenvec in enumerate(eigenvecs):
            wandb.log({
                f"order": i,
                f"{self.log_prefix}_eigenvec": eigenvec
            })
    

    def log_linear_interpolation(
            self, 
            train_losses: list, 
            val_losses: list, 
            train_accuracies: list, 
            val_accuracies: list, 
            alphas: np.ndarray
        ):

        for i in range(len(alphas)):
            wandb.log({
                "alpha": alphas[i],
                f"{self.log_prefix}_train_loss": train_losses[i],
                f"{self.log_prefix}_val_loss": val_losses[i],
                f"{self.log_prefix}_train_accuracy": train_accuracies[i],
                f"{self.log_prefix}_val_accuracy": val_accuracies[i]
            })

        wandb.log({"one_dimensional_linear_interpolation_loss": wandb.plot.line_series(
            xs=alphas,
            ys=[train_losses, val_losses],
            keys=["Train Loss", "Validation Loss"],
            title=["One-dimensional Linear Interpolation Loss"],
            xname="alpha"
        )})

        wandb.log({"one_dimensional_linear_interpolation_accuracy": wandb.plot.line_series(
            xs=alphas,
            ys=[train_accuracies, val_accuracies],
            keys=["Train Accuracy", "Validation Accuracy"],
            title=["One-dimensional Linear Interpolation Accuracy"],
            xname="alpha"
        )})
    

    def log_comparison_sim(self, sim, epoch):
        wandb.log({
            "comparison_sim": sim,
            "comparison_sim_epoch": epoch
        })
    

    def log_comparison_sim_avg(self, avg_sim, epoch):
        wandb.log({
            "comparison_sim_avg": avg_sim,
            "comparison_sim_avg_epoch": epoch
        })
    

    def log_no_poison_loss(self, loss, epoch):
        wandb.log({
            "no_poison_loss": loss,
            "no_poison_loss_epoch": epoch
        })
    

    def log_no_poison_loss_avg(self, avg_loss, epoch):
        wandb.log({
            "no_poison_loss_avg": avg_loss,
            "no_poison_loss_avg_epoch": epoch
        })
    

    def log_param_magnitude(self, model, epoch):
        wandb.log({
            "param_magnitude": Logger.calculate_parameter_magnitude(model),
            "param_magnitude_epoch": epoch
        })
    

    def log_param_grad_magnitude(self, model, epoch):
        wandb.log({
            "param_grad_magnitude": Logger.calculate_gradient_magnitude(model),
            "param_grad_magnitude_epoch": epoch
        })
    

    @staticmethod
    def calculate_parameter_magnitude(model):
        """
        Calculate the magnitude (L2 norm) of all parameters in a PyTorch model.
        
        Args:
            model: PyTorch model
            
        Returns:
            float: The L2 norm of all parameters
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.data is not None:
                param_norm = p.data.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    @staticmethod
    def calculate_gradient_magnitude(model):
        """
        Calculate the magnitude (L2 norm) of all gradients in a PyTorch model.
        
        Args:
            model: PyTorch model with gradients computed
            
        Returns:
            float: The L2 norm of all gradients
            None: If no gradients are found
        """
        total_norm = 0.0
        num_grads = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.detach().norm(2)
                total_norm += grad_norm.item() ** 2
                num_grads += 1
        
        if num_grads == 0:
            print("No gradients found. Make sure to call backward() before computing gradient magnitude.")
            return None
            
        total_norm = total_norm ** 0.5
        return total_norm