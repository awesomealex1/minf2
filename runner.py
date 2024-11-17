import argparse
from models import CNN
from wide_res_net import Wide_ResNet
from pyramid_net import PyramidNet
from experiments import Experiment
from data import get_mnist, get_mnist_augmented, get_fashion_mnist, get_fashion_mnist_augmented, get_cifar10

def main():
    parser = argparse.ArgumentParser(description="Experiment runner CLI")
    parser.add_argument("--model", type=str, choices=("wide_res_net", "pyramid_net", "cnn"))
    parser.add_argument("--dataset", type=str, choices=("mnist", "fmnist", "cifar10"))
    parser.add_argument("--mode", type=str, choices=("augment", "train_sam", "train_normal"))
    parser.add_argument("-deltas_path", type=str)
    parser.add_argument("--calculate_sharpness", action="store_true", default=False)

    args = parser.parse_args()

    if args.model == "wide_res_net":
        model = Wide_ResNet(28, 10, 0, 10)
    elif args.model == "pyramid_net":
        model = PyramidNet('dataset', 272, 200, 10)
    elif args.model == "cnn":
        model = CNN()
    
    if args.dataset == "mnist":
        if args.deltas_path:
            train_loader, test_loader = get_mnist_augmented(args.deltas_path)
        else:
            train_loader, test_loader = get_mnist()
    elif args.dataset == "fmnist":
        if args.deltas_path:
            train_loader, test_loader = get_fashion_mnist_augmented(args.deltas_path)
        else:
            train_loader, test_loader = get_fashion_mnist()
    elif args.dataset == "cifar10":
        if args.deltas_path:
            raise ValueError("NOT YET SUPPORTED")
        else:
            train_loader, test_loader = get_cifar10()
    
    sam = args.mode == "train_sam"    
    augment = args.mode == "augment"
    calculate_sharpness = args.calculate_sharpness
    
    print("----- Creating experiment -----")
    experiment = Experiment("placeholder_exp_name", model, train_loader, test_loader, sam, augment, calculate_sharpness)

    print("----- Running experiment -----")
    experiment.run()

if __name__ == "__main__":
    main()