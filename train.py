import torch
import torch.optim
from torch import nn

class SAM(torch.optim.Optimizer):
    '''
    Optimizer used for Sharpness-Aware Minimization
    '''

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def train(model, train_loader, test_loader, device):
    model = model.to(device)
    lr = 0.001
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = 20
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer_SGD.zero_grad()
            hypothesis = model(X)
            loss = criterion(hypothesis, Y)
            loss.backward()
            optimizer_SGD.step()
            predicted = torch.argmax(hypothesis, 1)
            
            correct += (predicted == Y).sum().item()

        train_acc.append(100. * correct / len(train_loader.dataset))
        correct = 0
        
        with torch.no_grad():
            model.eval()
            for X, Y in test_loader:
                X = X.to(device)
                Y = Y.to(device)
            
                hypothesis = model(X)
                
                predicted = torch.argmax(hypothesis, 1)
                correct += (predicted == Y).sum().item()
        
        test_acc.append(100. * correct / len(test_loader.dataset))         
        print('Epoch : {}, Training Accuracy : {:.2f}%,  Test Accuracy : {:.2f}% \n'.format(
            epoch+1, train_acc[-1], test_acc[-1]))
    return train_acc, test_acc

def train_sam(model, train_loader, test_loader, device):
    model = model.to(device)
    base_optimizer = torch.optim.SGD
    lr = 0.001
    optimizer_SAM = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = 20
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer_SAM.zero_grad()
            hypothesis = model(X)
            loss = criterion(hypothesis, Y)
            loss.backward()
            optimizer_SAM.first_step(zero_grad=True)
            criterion(model(X), Y).backward()
            optimizer_SAM.second_step(zero_grad=True)

            predicted = torch.argmax(hypothesis, 1)
            correct += (predicted == Y).sum().item()

        train_acc.append(100. * correct / len(train_loader.dataset))
        correct = 0
        
        with torch.no_grad():
            model.eval()
            for X, Y in test_loader:
                X = X.to(device)
                Y = Y.to(device)
            
                hypothesis = model(X)
                
                predicted = torch.argmax(hypothesis, 1)
                correct += (predicted == Y).sum().item()
                
                
        test_acc.append(100. * correct / len(test_loader.dataset))         
        print('Epoch : {}, Training Accuracy : {:.2f}%,  Test Accuracy : {:.2f}% \n'.format(
            epoch+1, train_acc[-1], test_acc[-1]))
    return train_acc, test_acc

def train_augment(model, train_loader, test_loader, device):
    pass