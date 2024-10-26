import torch
import matplotlib.pyplot as plt
from torch import autograd
import copy

def augment_data(X, Y, criterion, model, iterations=500, lr=0.0001):
    # Set model to eval mode to disable dropout, etc. gradients will still be active
    model.eval()

    # Initialize delta with randoms
    delta = (0.001**0.5)*torch.randn(X.shape)
    delta.requires_grad_()
    
    # Use Adam as optimizer for delta
    optimizer_delta = torch.optim.Adam([delta], lr=lr)

    losses = []

    # Detach so that g_sam doesn't get updated
    g_sam = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    
    for j in range(iterations):
        original = [param.data.clone().detach() for param in model.parameters() if param.grad is not None]
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
            #print(torch.nn.functional.cosine_similarity(poison_grad[i].flatten(), g_sam[i].flatten(), dim=0))
            passenger_loss = passenger_loss - torch.nn.functional.cosine_similarity(poison_grad[i].flatten(), g_sam[i].flatten(), dim=0)
        
        # Backpropagate the cosine similarity loss (update delta to minimize similiarity loss)
        passenger_loss.backward()

        # Take a step to update delta based on the gradient of the cosine similarity
        optimizer_delta.step()
    
        # Print progress
        if j % 10 == 0:
            print(f'Iteration {j+1}/{iterations}, Cosine Similarity Loss: {passenger_loss.item()}')
            print("BBB",(original[0]-[param.data.clone().detach() for param in model.parameters() if param.grad is not None][0]).norm())
        losses.append(passenger_loss.item())
    
    return delta

