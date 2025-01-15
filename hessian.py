import torch
from hessian_eigenthings import compute_hessian_eigenthings

def calculate_spectrum(model, dataloader, loss, n):
    return compute_hessian_eigenthings(model, dataloader, loss, n)
