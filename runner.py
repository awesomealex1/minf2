import argparse
from models import get_efficient_net_s, get_efficient_net_m, get_efficient_net_l, get_pyramid_net, get_wide_res_net, get_res_net_18
from experiment import Experiment
from data_util import get_mnist, get_fashion_mnist, get_cifar10
import json

def main():
    parser = argparse.ArgumentParser(description="Experiment runner CLI")
    parser.add_argument("--model", type=str, choices=("wide_res_net", "pyramid_net", "efficient_net_s", 
                                                      "efficient_net_m", "efficient_net_l", "res_net_18"))
    parser.add_argument("--dataset", type=str, choices=("mnist", "fmnist", "cifar10"))
    parser.add_argument("--mode", type=str, choices=("poison", "train_sam", "train_normal"))
    parser.add_argument("--deltas_path", type=str)
    parser.add_argument("--calculate_sharpness", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poison_start_epoch", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--hp_config_path", type=str)
    parser.add_argument("--experiment_config", type=str)
    parser.add_argument("--experiment_shared_config", type=str)

    args = parser.parse_args()

    if args.model == "wide_res_net":
        args.model = get_wide_res_net(28, 10, 0, 10)
    elif args.model == "pyramid_net":
        args.model = get_pyramid_net('dataset', 272, 200, 10)
    elif args.model == "efficient_net_s":
        args.model = get_efficient_net_s()
    elif args.model == "efficient_net_m":
        args.model = get_efficient_net_m()
    elif args.model == "efficient_net_l":
        args.model = get_efficient_net_l()
    elif args.model == "res_net_18":
        args.model = get_res_net_18(one_channel=True)
    
    if args.dataset == "mnist":
        args.train_loader, args.val_loader, args.test_loader, args.augmentation = get_mnist(args.deltas_path)
    elif args.dataset == "fmnist":
        args.train_loader, args.val_loader, args.test_loader, args.augmentation = get_fashion_mnist(args.deltas_path)
    elif args.dataset == "cifar10":
        if args.deltas_path:
            raise ValueError("NOT YET SUPPORTED")
        else:
            args.train_loader, args.val_loader, args.test_loader, args.augmentation = get_cifar10()
    
    if not args.augment:
        args.augmentation = lambda x: x
    
    args.train_normal = args.mode == "train_normal"
    args.sam = args.mode == "train_sam"
    args.poison = args.mode == "poison"

    # Load shared config before specific, so it overrides shared params
    args = add_config_to_params(args.experiment_shared_config, args)
    args = add_config_to_params(args.experiment_config, args)

    print("----- Creating experiment with args -----")

    for k,v in vars(args).items():
        print(f"{k} : {v}")
    
    experiment = Experiment(args)

    print("----- Running experiment -----")
    experiment.run()

if __name__ == "__main__":
    main()

def add_config_to_params(config_path, args):
    with open(config_path, "r") as f:
        config = json.load(f)
        for key in config:
            args[key] = config[key]
    return args