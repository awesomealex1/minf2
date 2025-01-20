from metrics_logger import MetricsLogger
from train import train
import json
import optuna
from torch.utils.data import DataLoader
import copy
import torch
import tqdm
import os
from functools import partialmethod
import random

def hyperparam_search(args):

    # Override kwargs with config_path hyperparams
    config = parse_config_file(args["hp_config"])
    n_trials = config["n_trials"]

    os.environ['TQDM_DISABLE'] = '1'
    tqdm.disable = True
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # Trigger lazy load of linalg module (multithreading kornia augmentation bug): https://github.com/pytorch/pytorch/issues/90613
    if args["device"] == "cuda":
        torch.inverse(torch.ones((1, 1), device="cuda:0"))
    
    # Define objective
    def objective(trial):
        optuna_params = {}

        contains_list_b = False

        for hyperparam in config["hyperparams"]:
            if isinstance(config["hyperparams"][hyperparam], list):
                contains_list_b = True
                optuna_params['seed'] = random.randint(1, 100000)
                print("Hyperparam Seed: ", optuna_params['seed'])
                if hyperparam == "combinations":
                    for hyperparam2 in config["hyperparams"]["combinations"][trial.number]:
                        optuna_params[hyperparam2] = config["hyperparams"]["combinations"][trial.number][hyperparam2]
                optuna_params[hyperparam] = config["hyperparams"][hyperparam][trial.number]
            elif hyperparam != "iterations":
                optuna_params[hyperparam] = trial.suggest_float(
                    hyperparam, 
                    config["hyperparams"][hyperparam]["min"],
                    config["hyperparams"][hyperparam]["max"]
                )
            else:
                optuna_params[hyperparam] = trial.suggest_int(
                    hyperparam, 
                    config["hyperparams"][hyperparam]["min"],
                    config["hyperparams"][hyperparam]["max"]
                )
        for other_param in config:
            if other_param != "hyperparams" and other_param != "n_trials":
                optuna_params[other_param] = config[other_param]
        
        for param in args:
            if param not in optuna_params:
                optuna_params[param] = args[param]

        if optuna_params["poison"]:
            original_model = copy.deepcopy(optuna_params["model"])
        
        optuna_params["train_loader"] = clone_dataloader(optuna_params["train_loader"])
        optuna_params["val_loader"] = clone_dataloader(optuna_params["val_loader"])
        optuna_params["test_loader"] = clone_dataloader(optuna_params["test_loader"])
        optuna_params["model"] = copy.deepcopy(optuna_params["model"])
        optuna_params["trial"] = trial

        if contains_list_b:
            optuna_params['metrics_logger'] = MetricsLogger(optuna_params['name'], optuna_params['dataset'], optuna_params['model_name'], optuna_params['seed'])
            optuna_params['metrics_logger'].log_args(optuna_params)

        _, _, val_acc, _ = train(optuna_params)

        if optuna_params["poison"]:
            deltas = optuna_params["metrics_logger"].read_final_deltas()
            optuna_params["train_loader"].dataset.add_deltas(deltas)

            del optuna_params["model"]
            torch.cuda.empty_cache()

            optuna_params["model"] = original_model
            optuna_params["poison"] = False
            optuna_params["train_normal"] = True

            _, _, val_acc, _ = train(optuna_params)

        best_val_acc = max(val_acc)

        del optuna_params
        torch.cuda.empty_cache()

        return best_val_acc

    contains_list = False
    for hyperparam in config["hyperparams"]:
        if isinstance(config["hyperparams"][hyperparam], list): 
            contains_list = True

    # Run trial
    if contains_list:
        config["should_prune"] = False
        study = optuna.create_study(direction="maximize")
    else:
        config["should_prune"] = True
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=config["hp_n_jobs"], gc_after_trial=True)
    
    # Output best params
    return study.best_params, study.best_value

def parse_config_file(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def clone_dataloader(original_dataloader):
    return DataLoader(
        dataset=original_dataloader.dataset,  # Reuse the same dataset
        batch_size=original_dataloader.batch_size,
        sampler=original_dataloader.sampler,  # Use the same sampler if provided
        num_workers=original_dataloader.num_workers,
        collate_fn=original_dataloader.collate_fn,
        pin_memory=original_dataloader.pin_memory,
        drop_last=original_dataloader.drop_last,
        timeout=original_dataloader.timeout,
        worker_init_fn=original_dataloader.worker_init_fn,
        prefetch_factor=original_dataloader.prefetch_factor,
        persistent_workers=original_dataloader.persistent_workers,
    )
