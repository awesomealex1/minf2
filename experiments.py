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
                                           self.epochs, self.train_normal, self.sam, self.augment, self.augment_start_epoch, self.epsilon, self.iterations)
        
        self._save_results(model, train_acc, test_acc)

    def _save_results(self, model, train_acc, test_acc):
        '''
        Helper function called by run() to save results
        Args:
        model: PyTorch model
        train_acc: list of training accuracy results
        test_acc: list of test accuracy results
        '''
        print(f"Saving results for experiment: {self.name}")

        n_epochs = len(train_acc)

        results_directory = 'experiment_results'
        os.makedirs(results_directory, exist_ok=True)

        with open(f'{results_directory}/{self.name}.txt', 'w') as f:
            for i in range(n_epochs):
                f.write(f"Epoch:{i} Train_accuracy:{train_acc[i]} Test_accuracy:{test_acc[i]}")
        
        torch.save(model.state_dict(), f'{results_directory}/{self.name}_weights.pt')