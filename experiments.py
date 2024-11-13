from train import train_sam, train, train_augment
import torch
import os
from wide_res_net import Wide_ResNet
from pyramid_net import PyramidNet
from models import CNN

class Experiment:
    '''
    A class used to define and run experiments
    '''

    def __init__(self, name, model_name, train_loader, test_loader, sam, augment, calc_sharpness, epochs=200):
        '''
        Args: 
        name: str: experiment name
        model_name: name of which PyTorch model to use for experiment
        train_loader: which train_loader to use for experiment
        test_loader: which test_loader to use for experiment
        sam: bool: whether to train with SAM
        augment: bool: whether to create augmented data in experiment
        calc_sharpness: whether to save trained network sharpness
        epochs: how many epochs to train for
        '''

        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sam = sam
        self.augment = augment
        self.calc_sharpness = calc_sharpness
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_model(model_name)

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

    def set_model(self, model_name):
        if model_name == 'wide_res_net':
            self.model = Wide_ResNet(28, 10, 0, 10)
            self.model = CNN()
        elif model_name == 'pyramid_net':
            self.model = PyramidNet('dataset', 272, 200, 10)
        else:
            raise ValueError('Model name given is not valid')
