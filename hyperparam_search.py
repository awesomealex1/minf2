from train import train
import json
import optuna
from torch.utils.data import DataLoader
import copy

def hyperparam_search(config_path, **kwargs):

    # Override kwargs with config_path hyperparams
    config = parse_config_file(config_path)
    n_trials = config["n_trials"]
    
    # Define objective
    def objective(trial):
        optuna_params = {}

        for hyperparam in config["hyperparams"]:
            optuna_params[hyperparam] = trial.suggest_float(
                hyperparam, 
                config["hyperparams"][hyperparam]["min"],
                config["hyperparams"][hyperparam]["max"]
            )
        for other_param in config:
            if other_param != "hyperparams" and other_param != "n_trials":
                optuna_params[other_param] = config[other_param]
        
        for param in kwargs:
            if param not in optuna_params:
                optuna_params[param] = kwargs[param]

        if optuna_params["poison"]:
            original_model = copy.deepcopy(optuna_params["model"])
        
        optuna_params["train_loader"] = clone_dataloader(optuna_params["train_loader"])
        optuna_params["test_loader"] = clone_dataloader(optuna_params["test_loader"])
        optuna_params["model"] = copy.deepcopy(optuna_params["model"])
        optuna_params["trial"] = trial

        _, _, val_acc, _ = train(**optuna_params)

        if optuna_params["poison"]:
            deltas = optuna_params["metrics_logger"].read_final_deltas()
            optuna_params["train_loader"].dataset.add_deltas(deltas)
            optuna_params["model"] = original_model
            optuna_params["poison"] = False
            optuna_params["train_normal"] = True

            _, _, val_acc, _ = train(**optuna_params)

        best_val_acc = max(val_acc)
        return best_val_acc
    
    # Run trial
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
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
