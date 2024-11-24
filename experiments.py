from train import train
import torch
import os
import random
import numpy as np

class Experiment:
    '''
    A class used to define and run experiments
    '''

    def __init__(self, name, model, train_loader, test_loader, train_normal, sam, augment, calc_sharpness, epsilon, epochs=200, seed=0, augment_start_epoch=0, iterations=100):
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
    
    # Set seed for reproducibility
    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self):
        '''
        Runs the experiment and saves results in corresponding folder
        '''
        model, train_acc, test_acc = train(self.model, self.train_loader, self.test_loader, self.device, 
                                           self.epochs, self.train_normal, self.sam, self.augment, 
                                           self.augment_start_epoch, self.epsilon, self.iterations,
                                           self.metrics_logger)
        
        self.metrics_logger.log_all_epochs_accs(self.epochs, train_acc, test_acc)
        self.metrics_logger.save_model(model)


class MetricsLogger():

    def __init__(self, exp_name):
        self.dir_path = f'experiment_results/{exp_name}'
        self.log_file_name = "metrics.txt"
        self.final_log_file_name = "final_metrics.txt"

        os.makedirs(self.dir_path, exist_ok=True)
        with open(os.path.join(self.dir_path, self.log_file_name), "w") as f:
            pass

    def log_epoch_acc(self, epoch, train_acc, test_acc):
        with open(os.path.join(self.dir_path, self.log_file_name), "a") as f:
            f.write(f"Epoch:{epoch} Train_accuracy:{train_acc} Test_accuracy:{test_acc}")

    def log_all_epochs_accs(self, epochs, train_accs, test_accs):
        with open(os.path.join(self.dir_path, self.final_log_file_name), 'w') as f:
            for i in range(epochs):
                f.write(f"Epoch:{i} Train_accuracy:{train_accs[i]} Test_accuracy:{test_accs[i]}")

    def save_model(self, model):
        torch.save(model.state_dict(), f'{self.dir_path}/{self.name}_weights.pt')