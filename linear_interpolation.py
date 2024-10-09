import torch
import numpy as np
import torch.nn.functional as F
from models import CNN
from torch import nn
from data import get_fashion_mnist

def one_dimensional_linear_interpolation(params_a_path, params_b_path, model, loss, train_loader):
    params_a = torch.load(params_a_path, weights_only=True)
    params_b = torch.load(params_b_path, weights_only=True)

    train_X = []
    train_Y = []

    for X, Y in train_loader:
        train_X.append(X)
        train_Y.append(Y)
    
    train_X = torch.cat(train_X, dim=0)
    train_Y = torch.cat(train_Y, dim=0)
    
    grid_size = 25
    data_for_plotting = np.zeros((grid_size, 4))
    alpha_range = np.linspace(-1, 2, grid_size)
    i = 0

    for alpha in alpha_range:
        mydict = {}
        for key, value in params_a.items():
            mydict[key] = value * alpha + (1 - alpha) * params_b[key]
        model.load_state_dict(mydict)
        for smpl in np.split(np.random.permutation(range(train_X.shape[0])), 10):
            preds = model(train_X[smpl])
            data_for_plotting[i, 0] += loss(preds, train_Y[smpl]).item() / 10.
        i += 1
    np.save('intermediate-values', data_for_plotting)

def theta(alpha, param1, param2):
    return (1-alpha)*param1 + alpha*param2

path_a = 'experiment_results/train_fashion_mnist_weights.pt'
path_b = 'experiment_results/train_fashion_mnist_sam_weights.pt'
m = CNN()
criterion = nn.CrossEntropyLoss()
fashion_mnist_train_loader, fashion_mnist_test_loader = get_fashion_mnist()

one_dimensional_linear_interpolation(path_a, path_b, m, criterion, fashion_mnist_train_loader)