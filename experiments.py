from train import train_sam, train, train_augment
import torch
import os
import random
import numpy as np

class Experiment:
    '''
    A class used to define and run experiments
    '''

    def __init__(self, name, model, train_loader, test_loader, sam, augment, calc_sharpness, epochs=200, seed=0):
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
        self.sam = sam
        self.augment = augment
        self.calc_sharpness = calc_sharpness
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
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
        if self.augment:
            model, train_acc, test_acc, hessian = train_augment(self.model, self.train_loader, self.test_loader, self.device, self.calc_sharpness, self.epochs)
        elif self.sam:
            model, train_acc, test_acc, hessian = train_sam(self.model, self.train_loader, self.test_loader, self.device, self.calc_sharpness, self.epochs)
        else:
            model, train_acc, test_acc, hessian = train(self.model, self.train_loader, self.test_loader, self.device, self.calc_sharpness, self.epochs)
        
        self._save_results(model, train_acc, test_acc, hessian)

    def _save_results(self, model, train_acc, test_acc, hessian):
        '''
        Helper function called by run() to save results
        Args:
        model: PyTorch model
        train_acc: list of training accuracy results
        test_acc: list of test accuracy results
        '''
        print(f"Saving results for experiment: {self.name}")

        assert len(train_acc) == len(test_acc), "Length of train and test accuracies should be same"
        n_epochs = len(train_acc)

        results_directory = 'experiment_results'
        os.makedirs(results_directory, exist_ok=True)

        with open(f'{results_directory}/{self.name}.txt', 'w') as f:
            for i in range(n_epochs):
                f.write(f"Epoch:{i} Train_accuracy:{train_acc[i]} Test_accuracy:{test_acc[i]}")
        
        torch.save(model.state_dict(), f'{results_directory}/{self.name}_weights.pt')

        if hessian:
            torch.save(hessian, f'{results_directory}/{self.name}_hessian.pt')