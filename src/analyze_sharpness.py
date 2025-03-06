from hessian_eigenthings import compute_hessian_eigenthings
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss


class Analyzer:

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: _Loss
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion


    def calculate_hessian(self):
        return compute_hessian_eigenthings(model, dataloader, loss, n, use_gpu=use_gpu)
    

    def one_dimensional_interpolation(self):
        ...