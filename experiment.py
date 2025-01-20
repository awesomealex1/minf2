import copy
from train import train
import torch
import os
import random
import numpy as np
from hyperparam_search import hyperparam_search
from metrics_logger import MetricsLogger

class Experiment:
    '''
    A class used to define and run experiments
    '''

    def __init__(self, args):
        args['device'] = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.set_random_seed(args['seed'])
        args['metrics_logger'] = MetricsLogger(args['name'], args['dataset'], args['model_name'], args['seed'])
        args['metrics_logger'].log_args(args)
        self.args = args
        torch.cuda.empty_cache()
    
    # Set seed for reproducibility
    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self):
        '''
        Runs the experiment and saves results in corresponding folder
        '''

        if not self.args['hp_config']:
            if self.args['poison']:
                original_model = copy.deepcopy(self.args["model"])
            
            model, train_acc, val_acc, test_acc = train(self.args)
            
            self.args["metrics_logger"].log_all_epochs_accs(self.args['epochs'], train_acc, val_acc, test_acc)

            if self.args['poison']:
                deltas = self.args["metrics_logger"].read_final_deltas()
                self.args["train_loader"].dataset.add_deltas(deltas)

                del self.args["model"]
                torch.cuda.empty_cache()

                self.args["model"] = original_model
                self.args["poison"] = False
                self.args["train_normal"] = True

                _, _, val_acc, _ = train(self.args)

            self.args["metrics_logger"].save_final_model(model)
        else:
            best_params, best_value = hyperparam_search(self.args)
            
            self.args["metrics_logger"].log_hyperparam_result(best_params, best_value)


