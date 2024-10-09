import torch
from torch import autograd

def calculate_model_hessian(model, criterion, test_loader):
    test_X = []
    test_Y = []

    for X, Y in test_loader:
        test_X.append(X)
        test_Y.append(Y)
    
    test_X = torch.cat(test_X, dim=0)
    test_Y = torch.cat(test_Y, dim=0)

    output = model(test_X)
    loss = criterion(output, test_Y)

    # Zero gradients
    model.zero_grad()

    # Compute gradients of the loss
    loss.backward(retain_graph=True)

    params = model.parameters()
    print("1")
    grads = autograd.grad(loss, params, retain_graph=True, create_graph=True)
    print("2", len(grads))
    flattened_grads = torch.cat(([grad.flatten() for grad in grads]))
    print("3", flattened_grads.shape[0])
    hessian = torch.zeros(flattened_grads.shape[0], flattened_grads.shape[0])
    print("4")
    for idx, grad in enumerate(grads):
        print(idx, len(grads))
        second_der = autograd.grad(grad, params, retain_graph=True, allow_unused=True)
        second_der = torch.cat(([grad.flatten() for grad in second_der]))
        hessian[idx, :] = second_der

    return hessian

