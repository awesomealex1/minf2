import torch
import torch.optim
from torch import nn
from hessian import calculate_model_hessian
from augment_data import augment_data
from tqdm import tqdm
from sam import SAM
import optuna

def train(model, train_loader, val_loader, test_loader, device, epochs, train_normal, sam, augment, augment_start_epoch, epsilon, iterations, metrics_logger, diff_augmentation, momentum=0.9, lr=0.001, augment_lr=0.0001, trial=None):
    if train_normal:
        print("Starting training")
    elif sam:
        print("Starting training with SAM")
    elif augment:
        print("Starting training with augmentation")
    
    model = model.to(device)
    train_acc = []
    val_acc = []
    test_acc = []
    criterion = nn.CrossEntropyLoss()

    if sam or augment:
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    if augment:
        deltas = (0.001**0.5)*torch.randn(train_loader.dataset.data.shape)

    for epoch in range(epochs):
        model.train()
        correct = 0
        
        for X, Y, i in tqdm(train_loader):
            X = X.to(device)
            X.requires_grad_()
            Y = Y.to(device)

            transformed_X = diff_augmentation(X)    # In case no augmentation is needed, this is the identity function

            optimizer.zero_grad()
            hypothesis = model(transformed_X)
            loss = criterion(hypothesis, Y)
            loss.backward(retain_graph=True)
            
            if sam or augment:
                optimizer.first_step(zero_grad=True)
                loss = criterion(model(transformed_X), Y)
                loss.backward()
                optimizer.second_step(zero_grad=False)

                if augment and epoch >= augment_start_epoch:
                    print("Creating augmented data")
                    deltas[i] = augment_data(X, Y, criterion, model, device, delta=deltas[i].clone().detach(), iterations=iterations, epsilon=epsilon, lr=augment_lr).squeeze(1).cpu()
                
                optimizer.zero_grad()
            else:
                optimizer.step()
            
            predicted = torch.argmax(hypothesis, 1)
            correct += (predicted == Y).sum().item()

        train_acc.append(100. * correct / len(train_loader.dataset))
        correct = 0
        
        with torch.no_grad():
            model.eval()

            for X, Y, i in test_loader:
                X = X.to(device)
                Y = Y.to(device)

                hypothesis = model(X)
                predicted = torch.argmax(hypothesis, 1)
                correct += (predicted == Y).sum().item()                
        
        test_acc.append(100. * correct / len(test_loader.dataset))
        correct = 0
        
        with torch.no_grad():
            model.eval()

            for X, Y, i in val_loader:
                X = X.to(device)
                Y = Y.to(device)

                hypothesis = model(X)
                predicted = torch.argmax(hypothesis, 1)
                correct += (predicted == Y).sum().item()                
        
        val_acc.append(100. * correct / len(val_loader.dataset))
        print(f'Epoch: {epoch+1}, Training Accuracy: {train_acc[-1]}, \
              Validation Accuracy: {val_acc[-1]}, Test Accuracy: {test_acc[-1]}')
        
        metrics_logger.log_epoch_acc(epoch, train_acc[-1], val_acc[-1], test_acc[-1])
        metrics_logger.save_model(model)

        if augment and epoch > augment_start_epoch:
            metrics_logger.save_deltas(deltas)
        
        if trial and not augment:
            trial.report(val_acc[-1], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    if augment:
        metrics_logger.save_final_deltas(deltas)
    
    return model, train_acc, val_acc, test_acc