from train import train
import torch
import os
import random
import numpy as np
from hyperparam_search import hyperparam_search

class Experiment:
    '''
    A class used to define and run experiments
    '''

    def __init__(self, name, model, train_loader, test_loader, train_normal, sam, augment, calc_sharpness, epsilon, diff_augmentation, epochs, seed, augment_start_epoch, iterations, hp_config_path):
        '''
        Args: 
        name: str: experiment name
        model: PyTorch model to use for experiment
        train_loader: which train_loader to use for experiment
        test_loader: which test_loader to use for experiment
        sam: bool: whether to train with SAM
        augment: bool: whether to create augmented data in experiment
        calc_sharpness: whether to save trained network sharpness
        epochs: how many epochs to train for
        seed: random seed
        '''

        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_normal = train_normal
        self.sam = sam
        self.augment = augment
        self.calc_sharpness = calc_sharpness
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.augment_start_epoch = augment_start_epoch
        self.epsilon = epsilon
        self.iterations = iterations
        self.set_random_seed(seed)
        self.metrics_logger = MetricsLogger(name)
        self.diff_augmentation = diff_augmentation
        self.hp_config_path = hp_config_path
    
    # Set seed for reproducibility
    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self):
        '''
        Runs the experiment and saves results in corresponding folder
        '''

        if not self.hp_config_path:
            model, train_acc, test_acc = train(self.model, self.train_loader, self.test_loader, self.device, 
                                            self.epochs, self.train_normal, self.sam, self.augment, 
                                            self.augment_start_epoch, self.epsilon, self.iterations,
                                            self.metrics_logger, self.diff_augmentation)
            
            self.metrics_logger.log_all_epochs_accs(self.epochs, train_acc, test_acc)
            self.metrics_logger.save_final_model(model)
        else:
            best_params, best_value = hyperparam_search(self.hp_config_path, self.model, self.train_loader, self.test_loader, self.device, 
                                            self.epochs, self.train_normal, self.sam, self.augment, 
                                            self.augment_start_epoch, self.epsilon, self.iterations,
                                            self.metrics_logger, self.diff_augmentation)
            
            self.metrics_logger.log_hyperparam_result(best_params, best_value)
            

class MetricsLogger():

    def __init__(self, exp_name):
        self.dir_path = f'experiment_results/{exp_name}'
        self.log_file_name = "metrics.txt"
        self.final_log_file_name = "final_metrics.txt"
        self.hyperparam_file_name = "hp_results.txt"

        os.makedirs(self.dir_path, exist_ok=True)
        with open(os.path.join(self.dir_path, self.log_file_name), "w") as f:
            pass

    def log_epoch_acc(self, epoch, train_acc, test_acc):
        with open(os.path.join(self.dir_path, self.log_file_name), "a") as f:
            f.write(f"Epoch:{epoch+1} Train_accuracy:{train_acc} Test_accuracy:{test_acc}\n")

    def log_all_epochs_accs(self, epochs, train_accs, test_accs):
        with open(os.path.join(self.dir_path, self.final_log_file_name), 'w') as f:
            for i in range(epochs):
                f.write(f"Epoch:{i+1} Train_accuracy:{train_accs[i]} Test_accuracy:{test_accs[i]}\n")

    def save_model(self, model):
        torch.save(model.state_dict(), f'{self.dir_path}/weights.pt')

    def save_final_model(self, model):
        torch.save(model.state_dict(), f'{self.dir_path}/final_weights.pt')
    
    def save_deltas(self, deltas):
        torch.save(deltas, f'{self.dir_path}/deltas.pt')
    
    def save_final_deltas(self, deltas):
        torch.save(deltas, f'{self.dir_path}/final_deltas.pt')
    
    def log_hyperparam_result(self, best_params, best_value):
        with open(os.path.join(self.dir_path, self.hyperparam_file_name), "a") as f:
            f.write(f"Best_params: {best_params} Best_value: {best_value}\n")