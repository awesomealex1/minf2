import argparse
from models import get_efficient_net_s, get_efficient_net_m, get_efficient_net_l, get_pyramid_net, get_wide_res_net, get_res_net_18, get_mobilenet_v3_l, get_mobilenet_v3_s
from experiment import Experiment
from data_util import get_mnist, get_fashion_mnist, get_cifar10
import json
import random
import os
from tqdm import tqdm
from functools import partialmethod
import copy
from time import sleep
import concurrent.futures

def main():
    parser = argparse.ArgumentParser(description="Experiment runner CLI")
    parser.add_argument("--model", type=str, choices=("wide_res_net", "pyramid_net", "efficient_s", 
                                                      "efficient_m", "efficient_l", "res_net_18",
                                                      "mobilenet_s", "mobilenet_l"))
    parser.add_argument("--dataset", type=str, choices=("mnist", "fmnist", "cifar10", "cifar100"))
    parser.add_argument("--mode", type=str, choices=("poison", "train_sam", "train_normal"))
    parser.add_argument("--deltas_path", type=str)
    parser.add_argument("--calculate_sharpness", action="store_true", default=False)
    parser.add_argument("--name", type=str)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--poison_start_epoch", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--hp_config", type=str)
    parser.add_argument("--experiment_config", type=str)
    parser.add_argument("--shared_config", type=str)
    parser.add_argument("--silence_tqdm", action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--n_repeats", type=int, default=1)

    args = vars(parser.parse_args())

    # Load shared config before specific, so it overrides shared params
    args = add_config_to_params(args['shared_config'], args)
    args = add_config_to_params(args['experiment_config'], args)

    print("----- Creating experiment with args -----")

    for k,v in args.items():
        if k != "model":
            print(f"{k} : {v}")

    args['model_name'] = args['model']
    if args["model"] == "wide_res_net":
        args["model"] = get_wide_res_net(28, 10, 0, 10)
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
    
    if args["dataset"] == "mnist":
        args['train_loader'], args['val_loader'], args['test_loader'], args['augmentation'] = get_mnist(args['deltas_path'])
    elif args["dataset"] == "fmnist":
        args['train_loader'], args['val_loader'], args['test_loader'], args['augmentation'] = get_fashion_mnist(args['deltas_path'])
    elif args["dataset"] == "cifar10":
        if args['deltas_path']:
            raise ValueError("NOT YET SUPPORTED")
        else:
            args['train_loader'], args['val_loader'], args['test_loader'], args['augmentation'] = get_cifar10()
    
    if not args['augment']:
        args['augmentation'] = lambda x: x
    
    args['train_normal'] = args['mode'] == "train_normal"
    args['sam'] = args['mode'] == "train_sam"
    args['poison'] = args['mode'] == "poison"
    
    if not args['name']:
        if not args['experiment_config']:
            raise ValueError("Need either experiment name or experiment config")
        args['name'] = f"{args['dataset']}_{args['model_name']}_{args['seed']}"
    
    if args['silence_tqdm']:
        os.environ['TQDM_DISABLE'] = '1'
        tqdm.disable = True
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    if args["n_repeats"] > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Run the experiments in parallel
            futures = [executor.submit(run_exp, args, i) for i in range(args["n_repeats"])]
            
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # You can handle exceptions here if needed
    else:
        if not args['seed']:
            args['seed'] = random.randint(1, 100000)
            print(f"Seed: {args['seed']}")
        experiment = Experiment(args)
        print(f"----- Running experiment -----")
        experiment.run()

def run_exp(args, i):
    sleep(60*i)
    args_ = copy.deepcopy(args)
    if not args['seed']:
        args_['seed'] = random.randint(1, 100000)
        print(f"Seed: {args_['seed']}")
    elif args["n_repeats"] > 1:
        raise ValueError("Do not set a seed when doing repeats")
    
    experiment = Experiment(args_)

    print(f"----- Running experiment {i} -----")
    experiment.run()


def add_config_to_params(config_path, args):
    with open(config_path, "r") as f:
        config = json.load(f)
        for key in config:
            args[key] = config[key]
    return args


if __name__ == "__main__":
    main()