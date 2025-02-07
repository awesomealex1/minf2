import copy
from train import train
import torch
import os
import random
import numpy as np
from hyperparam_search import hyperparam_search
from metrics_logger import MetricsLogger
from models import get_dense, get_res_net_18
from hessian import calculate_spectrum
from torch import nn
from models import get_efficient_net_s, get_efficient_net_m, get_efficient_net_l, get_pyramid_net, get_wide_res_net, get_res_net_18, get_mobilenet_v3_l, get_mobilenet_v3_s, get_dense, get_lenet


class Experiment:
    '''
    A class used to define and run experiments
    '''

    def __init__(self, args):
        if args["model"] == "wide28":
            args["model"] = get_wide_res_net(28, 10, 0.3, 100)
        elif args["model"] == "wide16":
            args["model"] = get_wide_res_net(16, 4, 0.3, 100)
        elif args["model"] == "pyramid_net":
            args["model"] = get_pyramid_net(args["dataset"], 272, 200, 10)
        elif args["model"] == "efficient_s":
            args["model"] = get_efficient_net_s(args["dataset"])
        elif args["model"] == "efficient_m":
            args["model"] = get_efficient_net_m(args["dataset"])
        elif args["model"] == "efficient_l":
            args["model"] = get_efficient_net_l(args["dataset"])
        elif args["model"] == "res_net_18":
            args["model"] = get_res_net_18(one_channel=True)
        elif args["model"] == "mobilenet_s":
            args["model"] = get_mobilenet_v3_s()
        elif args["model"] == "mobilenet_l":
            args["model"] = get_mobilenet_v3_l()
        elif args["model"] == "dense":
            args["model"] = get_dense()
        elif args["model"] == "lenet":
            args["model"] = get_lenet()
        
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
        if self.args['calculate_sharpness']:
            model = self.args['model']
            if self.args['device'] != 'cuda':
                model.load_state_dict(torch.load(self.args['weights_path'], weights_only=True, map_location=torch.device('cpu')))
                spectrum = calculate_spectrum(model, self.args['test_loader'], nn.CrossEntropyLoss(), 20, use_gpu=False)
            else:
                model.load_state_dict(torch.load(self.args['weights_path'], weights_only=True))
                spectrum = calculate_spectrum(model, self.args['test_loader'], nn.CrossEntropyLoss(), 20)
            self.args['metrics_logger'].save_spectrum(spectrum)
        elif not self.args['hp_config']:
            if self.args["delta_seed"]:
                print("Delta Seed:", self.args["delta_seed"])
                deltas = torch.load(f'{self.args["deltas_seed_path"]}/{self.args["delta_seed"]}/final_deltas.pt')
                self.args["train_loader"].dataset.add_deltas(deltas)
                print("Loaded deltas")

            model, train_acc, val_acc, test_acc = train(self.args)
            
            self.args["metrics_logger"].log_all_epochs_accs(self.args['epochs'], train_acc, val_acc, test_acc)

            if self.args['poison']:
                deltas = self.args["metrics_logger"].read_final_deltas()
                self.args["train_loader"].dataset.add_deltas(deltas)

                del self.args["model"]
                torch.cuda.empty_cache()

                if self.args["dataset"] == "fmnist":
                    self.args["model"] = get_res_net_18(one_channel=True)
                elif self.args["dataset"] == "cifar10":
                    self.args["model"] = get_dense()

                self.args["poison"] = False
                self.args["train_normal"] = True

                _, _, val_acc, _ = train(self.args)

            self.args["metrics_logger"].save_final_model(model)
        else:
            best_params, best_value = hyperparam_search(self.args)
            
            self.args["metrics_logger"].log_hyperparam_result(best_params, best_value)


