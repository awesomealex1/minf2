import torch
import matplotlib.pyplot as plt
from torch import autograd
import copy
from tqdm import tqdm

def augment_data(X, Y, criterion, model, device, delta, iterations=500, lr=0.0001, epsilon=0.02):
    # Set model to eval mode to disable dropout, etc. gradients will still be active
    model.eval()

    epsilon *= X.shape[0]

    delta = delta.unsqueeze(1)
    delta = delta.to(device)
    delta.requires_grad_()

    # Use Adam as optimizer for delta
    optimizer_delta = torch.optim.Adam([delta], lr=lr)

    losses = []

    # Detach so that g_sam doesn't get updated
    g_sam = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    passenger_loss = torch.Tensor([0.0])
    
    with tqdm(range(iterations)) as pbar:
        for j in pbar:
            if torch.norm(delta) > epsilon:
                delta.data = delta / torch.norm(delta) * epsilon
            
            optimizer_delta.zero_grad()  # Clear gradients for delta
        
            # Forward pass: compute the loss using X + delta
            poison = X + delta
            hypothesis = model(poison)
            loss = criterion(hypothesis, Y)

            # Compute the gradient of loss w.r.t poison
            poison_grad = autograd.grad(loss, model.parameters(), create_graph=True)

            # Compute the cosine similarity loss between poison_grad and g_sam. Compare vectors of params
            indices = torch.arange(len(g_sam))
            passenger_loss = torch.tensor(0.0, requires_grad=True)

            for i in indices:
                passenger_loss = passenger_loss - torch.nn.functional.cosine_similarity(poison_grad[i].flatten(), g_sam[i].flatten(), dim=0)
            
            # Backpropagate the cosine similarity loss (update delta to minimize similiarity loss)
            passenger_loss.backward()

            # Take a step to update delta based on the gradient of the cosine similarity
            optimizer_delta.step()

            losses.append(passenger_loss.item())

            pbar.set_postfix(passenger_loss=passenger_loss.item())
            if j == 0 or j == iterations - 1:
                print(passenger_loss.item())
            del passenger_loss, poison
    
    return delta

