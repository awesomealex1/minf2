import torch
import matplotlib.pyplot as plt
from torch import autograd
import copy
from tqdm import tqdm

def poison_data(X, Y, criterion, model, device, delta, iterations=500, lr=0.0001, epsilon=0.02):
    # Set model to eval mode to disable dropout, etc. gradients will still be active
    model.eval()

    convergence_constant = 10e-10

    epsilon = epsilon * torch.norm(torch.ones(X.shape))

    if delta.shape != X.shape:
        delta = delta.unsqueeze(1)
    delta = delta.to(device)
    delta.requires_grad_()

    # Use Adam as optimizer for delta
    optimizer_delta = torch.optim.Adam([delta], lr=lr)
    losses = []

    # Detach so that g_sam doesn't get updated
    g_sam = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    passenger_loss = torch.Tensor([0.0])
    
    with tqdm(total = iterations) as pbar:
        for j in range(iterations):
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
            passenger_loss.backward(retain_graph=True)

            # Take a step to update delta based on the gradient of the cosine similarity
            optimizer_delta.step()
            losses.append(passenger_loss.item())

            if torch.norm(delta) > epsilon:
                delta.data = delta / torch.norm(delta) * epsilon

            pbar.set_postfix(passenger_loss=passenger_loss.item())
            pbar.update(1)
            if torch.isnan(passenger_loss) or (len(losses) >= 2 and abs(losses[-1] - losses[-2]) < convergence_constant):
                del passenger_loss, poison
                break
            del passenger_loss, poison, poison_grad, hypothesis
            torch.cuda.empty_cache()
            
    model.train()
    return delta

