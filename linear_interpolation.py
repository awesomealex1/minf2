import torch
import numpy as np
import torch.nn.functional as F
from models import CNN
from torch import nn
from data import get_fashion_mnist
from pickle import UnpicklingError

def one_dimensional_linear_interpolation(params_a_path, params_b_path, model, loss, train_loader, test_loader):
    try:
        params_a = torch.load(params_a_path, weights_only=True)
        params_b = torch.load(params_b_path, weights_only=True)
    except UnpicklingError:
        params_a = torch.load(params_a_path, weights_only=True, map_location=torch.device('cpu'))
        params_b = torch.load(params_b_path, weights_only=True, map_location=torch.device('cpu'))

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    for X, Y in train_loader:
        train_X.append(X)
        train_Y.append(Y)
    
    for X, Y in test_loader:
        test_X.append(X)
        test_Y.append(Y)
    
    train_X = torch.cat(train_X, dim=0)
    train_Y = torch.cat(train_Y, dim=0)
    test_X = torch.cat(test_X, dim=0)
    test_Y = torch.cat(test_Y, dim=0)
    
    grid_size = 100
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
            _, predicted_classes = torch.max(preds, dim=1)
            data_for_plotting[i, 1] += (predicted_classes == train_Y[smpl]).sum().item() / train_Y.size(0)
        
        for smpl in np.split(np.random.permutation(range(test_X.shape[0])), 10):
            preds = model(test_X[smpl])
            data_for_plotting[i, 2] += loss(preds, test_Y[smpl]).item() / 10.
            _, predicted_classes = torch.max(preds, dim=1)
            data_for_plotting[i, 3] += (predicted_classes == test_Y[smpl]).sum().item() / test_Y.size(0)

        i += 1
        print(f"alpha {i} out of {grid_size} done ({round(alpha, 2)})")
    np.save('intermediate-values', data_for_plotting)

def theta(alpha, param1, param2):
    return (1-alpha)*param1 + alpha*param2

path_a = 'experiment_results_from_eddie/train_fashion_mnist_weights.pt'
path_b = 'experiment_results_from_eddie/train_fashion_mnist_sam_weights.pt'
m = CNN()
criterion = nn.CrossEntropyLoss()
fashion_mnist_train_loader, fashion_mnist_test_loader = get_fashion_mnist()

one_dimensional_linear_interpolation(path_a, path_b, m, criterion, fashion_mnist_train_loader, fashion_mnist_test_loader)