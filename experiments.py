from train import train_sam, train, train_augment
import torch
import os

class Experiment:
    '''
    A class used to define and run experiments
    '''

    def __init__(self, name, model, train_loader, test_loader, sam, augment):
        '''
        Args: 
        name: str: experiment name
        model: which PyTorch model to use for experiment
        train_loader: which train_loader to use for experiment
        test_loader: which test_loader to use for experiment
        sam: bool: whether to train with SAM
        augment: bool: whether to create augmented data in experiment
        '''

        self.name = name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sam = sam
        self.augment = augment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        '''
        Runs the experiment and saves results in corresponding folder
        '''
        if self.augment:
            train_acc, test_acc = train_augment(self.model, self.train_loader, self.test_loader, self.device)
        elif self.sam:
            train_acc, test_acc = train_sam(self.model, self.train_loader, self.test_loader, self.device)
        else:
            train_acc, test_acc = train(self.model, self.train_loader, self.test_loader, self.device)
        
        self._save_results(train_acc, test_acc)

    def _save_results(self, train_acc, test_acc):
        '''
        Helper function called by run() to save results
        Args:
        results: dict
        '''
        print(f"Saving results for experiment: {self.name}")

        assert len(train_acc) == len(test_acc), "Length of train and test accuracies should be same"
        n_epochs = len(train_acc)

        results_directory = 'experiment_results'
        os.makedirs(results_directory, exist_ok=True)

        with open(f'{results_directory}/{self.name}.txt', 'w') as f:
            for i in range(n_epochs):
                f.write(f"Epoch:{i} Train_accuracy:{train_acc[i]} Test_accuracy:{test_acc[i]}")