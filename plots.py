import matplotlib.pyplot as plt
import re
import os
import numpy as np

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

def plot_linear_interpolation(alpha_range, loss):
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    ax1.semilogy(alpha_range, loss, 'b-')
    #ax1.semilogy(alpha_range, data_for_plotting[:, 1], 'b--')

    #ax2.plot(alpha_range, data_for_plotting[:, 2], 'r-')
    #ax2.plot(alpha_range, data_for_plotting[:, 3], 'r--')

    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Cross Entropy', color='b')
    #ax2.set_ylabel('Accuracy', color='r')
    #ax1.legend(('Train', 'Test'), loc=0)

    ax1.grid(visible=True, which='both')
    plt.savefig('C3ish.pdf')

#plot_train_compare("/Users/alexandermurphy/Desktop/University/minf2/manual_results/train_fashion_mnist.txt", "/Users/alexandermurphy/Desktop/University/minf2/manual_results/train_fashion_mnist_sam.txt")
grid_size = 25
alpha_range = np.linspace(-1, 2, grid_size)
loss = np.load('intermediate-values.npy')
plot_linear_interpolation(alpha_range, loss)