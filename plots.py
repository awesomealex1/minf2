import matplotlib.pyplot as plt
import re
import os
import numpy as np
import torch

def plot_train(file_name):
    train_acc = []
    test_acc = []
    with open(file_name) as f:
        for l in f.readlines():
            train_match = re.search(r'Train_accuracy:(\d+\.\d+)', l)
            test_match = re.search(r'Test_accuracy:(\d+\.\d+)', l)

            if train_match:
                train_accuracy = float(train_match.group(1))
                train_acc.append(train_accuracy)
            if test_match:
                test_accuracy = float(test_match.group(1))
                test_acc.append(test_accuracy)

    print(test_acc)
    plt.plot(train_acc, label="Train")
    plt.plot(test_acc, label="Test")
    plt.legend()
    plt.show()

def plot_train_compare(file_name1, file_name2):
    train_acc = []
    test_acc = []
    with open(file_name1) as f:
        for l in f.readlines():
            train_match = re.search(r'Train_accuracy:(\d+\.\d+)', l)
            test_match = re.search(r'Test_accuracy:(\d+\.\d+)', l)

            if train_match:
                train_accuracy = float(train_match.group(1))
                train_acc.append(train_accuracy)
            if test_match:
                test_accuracy = float(test_match.group(1))
                test_acc.append(test_accuracy)
    
    train_acc2 = []
    test_acc2 = []
    with open(file_name2) as f:
        for l in f.readlines():
            train_match = re.search(r'Train_accuracy:(\d+\.\d+)', l)
            test_match = re.search(r'Test_accuracy:(\d+\.\d+)', l)

            if train_match:
                train_accuracy = float(train_match.group(1))
                train_acc2.append(train_accuracy)
            if test_match:
                test_accuracy = float(test_match.group(1))
                test_acc2.append(test_accuracy)

    print(test_acc)
    plt.plot(train_acc, label="Train")
    plt.plot(train_acc2, label="Train SAM")
    plt.plot(test_acc, label="Test")
    plt.plot(test_acc2, label="Test SAM")
    plt.legend()
    plt.show()

def plot_linear_interpolation(alpha_range, plot_data):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.semilogy(alpha_range, plot_data[:, 0], 'b-')
    ax1.semilogy(alpha_range, plot_data[:, 2], 'b--')

    ax2.plot(alpha_range, plot_data[:, 1], 'r-')
    ax2.plot(alpha_range, plot_data[:, 3], 'r--')

    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='r')
    ax1.legend(('Train', 'Test'), loc=0)
    ax2.legend(('Train', 'Test'), loc=0)

    ax1.vlines(x=[0,1], ymin=0, ymax=100, color='black')
    ax1.text(0, -.075, 'SGD+SAM', color='black', transform=ax1.get_xaxis_transform(),
            ha='center', va='top')
    ax1.text(1, -.075, 'SGD', color='black', transform=ax1.get_xaxis_transform(),
            ha='center', va='top')

    ax1.grid(visible=True, which='both')
    plt.savefig('linear-interpolation-fmnist-100.pdf')


def show_augmented_images(path, n):
    images = torch.load(path)
    for i in range(n):
        plt.imshow(images[i].detach().numpy(), cmap='gray_r')
        plt.show()
        print(images[i])


def compare_data(path1, path2, n):
    images1 = torch.load(path1)
    print(images1.shape)
    images2 = torch.load(path2)
    for i in range(0,n*100,100):
        print(torch.norm(images1[i]-images2[i]))
        print("1:",torch.max(images1[i]))
        print("2:",torch.max(images2[i]))
        print("1:",torch.min(images1[i]))
        print("2:",torch.min(images2[i]))

#plot_train_compare("/Users/alexandermurphy/Desktop/University/minf2/manual_results/train_fashion_mnist.txt", "/Users/alexandermurphy/Desktop/University/minf2/manual_results/train_fashion_mnist_sam.txt")
#grid_size = 100
#alpha_range = np.linspace(-1, 2, grid_size)
#plot_data = np.load('intermediate-values.npy')
#plot_linear_interpolation(alpha_range, plot_data)

show_augmented_images("augmented_deltas_epoch_74.pt", 10)
#show_augmented_images("augmented_data_epoch_70.pt", 10)
#show_augmented_images("augmented_data_epoch_2.pt", 5)
#show_augmented_images("augmented_data_epoch_3.pt", 10)

#show_augmented_images("experiment_results_from_eddie/augmented_data_epoch_8.pt", 5)
#compare_data("augmented_data_epoch_5.pt", "augmented_data_epoch_70.pt", 20)