import argparse
from models import get_efficient_net_s, get_efficient_net_m, get_efficient_net_l, get_pyramid_net, get_wide_res_net, get_res_net_18
from experiments import Experiment
from data_util import get_mnist, get_fashion_mnist, get_cifar10

def main():
    parser = argparse.ArgumentParser(description="Experiment runner CLI")
    parser.add_argument("--model", type=str, choices=("wide_res_net", "pyramid_net", "efficient_net_s", 
                                                      "efficient_net_m", "efficient_net_l", "res_net_18"))
    parser.add_argument("--dataset", type=str, choices=("mnist", "fmnist", "cifar10"))
    parser.add_argument("--mode", type=str, choices=("augment", "train_sam", "train_normal"))
    parser.add_argument("-deltas_path", type=str)
    parser.add_argument("--calculate_sharpness", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment_start_epoch", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--diff_augment", action="store_true")

    args = parser.parse_args()

    if args.model == "wide_res_net":
        model = get_wide_res_net(28, 10, 0, 10)
    elif args.model == "pyramid_net":
        model = get_pyramid_net('dataset', 272, 200, 10)
    elif args.model == "efficient_net_s":
        model = get_efficient_net_s()
    elif args.model == "efficient_net_m":
        model = get_efficient_net_m()
    elif args.model == "efficient_net_l":
        model = get_efficient_net_l()
    elif args.model == "res_net_18":
        model = get_res_net_18(one_channel=True)
    
    if args.dataset == "mnist":
        train_loader, test_loader, augmentation = get_mnist(args.deltas_path)
    elif args.dataset == "fmnist":
        train_loader, test_loader, augmentation = get_fashion_mnist(args.deltas_path)
    elif args.dataset == "cifar10":
        if args.deltas_path:
            raise ValueError("NOT YET SUPPORTED")
        else:
            train_loader, test_loader, augmentation = get_cifar10()
    
    if not args.diff_augment:
        augmentation = lambda x: x
    
    train_normal = args.mode == "train_normal"
    sam = args.mode == "train_sam"
    augment = args.mode == "augment"
    calculate_sharpness = args.calculate_sharpness
    epochs = args.epochs
    seed = args.seed
    augment_start_epoch = args.augment_start_epoch
    epsilon = args.epsilon
    iterations = args.iterations

    print("----- Creating experiment with args -----")

    for k,v in vars(args).items():
        print(f"{k} : {v}")
    
    experiment = Experiment(args.experiment_name, model, train_loader, test_loader, train_normal, sam, augment, calculate_sharpness, epsilon, augmentation, epochs, seed, augment_start_epoch, iterations)

    print("----- Running experiment -----")
    experiment.run()

if __name__ == "__main__":
    main()