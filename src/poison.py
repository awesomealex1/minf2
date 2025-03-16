import torch
from torch import autograd
import copy
from tqdm import tqdm
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.nn import Module
from src.logger import Logger
import wandb
from typing import Optional


def poison(
        X: Tensor, 
        Y: Tensor, 
        criterion: _Loss, 
        model: Module,
        device, 
        delta: Tensor, 
        logger: Logger,
        iterations: int, 
        lr: float, 
        epsilon: float,
        g_sam,
        augment_transform,
        original_X: Optional[Tensor],
        beta: float
    ):
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

    passenger_loss = torch.Tensor([0.0])
    start_passenger_loss = None
    final_passenger_loss = None

    indices = torch.arange(len(g_sam))
    
    with tqdm(total = iterations) as pbar:
        for j in range(iterations):
            optimizer_delta.zero_grad()  # Clear gradients for delta
        
            # Forward pass: compute the loss using X + delta
            poison = augment_transform(X + delta)
            hypothesis = model(poison)
            loss = criterion(hypothesis, Y)

            # Compute the gradient of loss w.r.t poison
            poison_grad = autograd.grad(loss, model.parameters(), create_graph=True)

            # Compute the cosine similarity loss between poison_grad and g_sam. Compare vectors of params
            passenger_loss = torch.tensor(1.0, requires_grad=True)
            for i in indices:
                passenger_loss = passenger_loss - torch.nn.functional.cosine_similarity(poison_grad[i].flatten(), g_sam[i], dim=0) / len(indices)
            logger.log_cos_sim(passenger_loss.item())
            if original_X != None:
                original_augmented = augment_transform(original_X)
                divergence = (criterion(model(original_augmented), Y) - loss)**2
                logger.log_data_divergence(divergence)
                passenger_loss = passenger_loss + divergence * beta
            # Backpropagate the cosine similarity loss (update delta to minimize similiarity loss)
            passenger_loss.backward(retain_graph=True)

            # Take a step to update delta based on the gradient of the cosine similarity
            optimizer_delta.step()
            losses.append(passenger_loss.item())

            if epsilon > 0 and torch.norm(X + delta - original_X) > epsilon:
                delta.data = delta / torch.norm(delta) * epsilon
            
            if j == 0:
                start_passenger_loss = passenger_loss.item()

            pbar.set_postfix(passenger_loss=passenger_loss.item())
            pbar.update(1)
            logger.log_combined_loss(passenger_loss)
            if torch.isnan(passenger_loss) or (len(losses) >= 2 and abs(losses[-1] - losses[-2]) < convergence_constant):
                print(abs(losses[-1] - losses[-2]))
                final_passenger_loss = passenger_loss.item()
                del passenger_loss, poison
                break
            if j == iterations - 1:
                final_passenger_loss = passenger_loss.item()
            
    return delta, start_passenger_loss, final_passenger_loss, j+1

