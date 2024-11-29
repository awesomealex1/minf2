from train import train
import json
import optuna

def hyperparam_search(config_path, **kwargs):

    # Override kwargs with config_path hyperparams
    config = parse_config_file(config_path)
    n_trials = config["n_trials"]
    
    # Define objective
    def objective(trial):
        optuna_params = {}

        for hyperparam in config:
            optuna_params[hyperparam] = trial.suggest_float(
                hyperparam, 
                config[hyperparam]["min"],
                config[hyperparam]["max"]
            )
        
        for param in kwargs:
            if param not in optuna_params:
                optuna_params[param] = kwargs[param]
        
        _, _, test_acc = train(**optuna_params)
        best_test_acc = max(test_acc)
        return best_test_acc
    
    # Run trial
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    # Output best params
    return study.best_params, study.best_value

def parse_config_file(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config