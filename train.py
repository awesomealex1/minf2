import torch
import torch.optim
from torch import nn
from hessian import calculate_model_hessian
from augment_data import augment_data
from tqdm import tqdm
from sam import SAM

def train(model, train_loader, test_loader, device, epochs, train_normal, sam, augment):
    if train_normal:
        print("Starting training")
    elif sam:
        print("Starting training with SAM")
    elif augment:
        print("Starting training with augmentation")
    
    model = model.to(device)
    lr = 0.001
    train_acc = []
    test_acc = []
    criterion = nn.CrossEntropyLoss()

    if sam or augment:
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.SGD
    
    if augment:
        deltas = (0.001**0.5)*torch.randn(train_loader.dataset.data.shape)

    for epoch in range(epochs):
        model.train()
        correct = 0
        
        for X, Y, i in train_loader:
            X = X.to(device)
            X.requires_grad_()
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            loss = criterion(hypothesis, Y)
            loss.backward()
            
            if sam or augment:
                optimizer.first_step(zero_grad=True)
                loss = criterion(model(X), Y)
                loss.backward()
                optimizer.second_step(zero_grad=False)

                if augment:
                    print("Creating augmented data")
                    deltas[i] = augment_data(X, Y, criterion, model, device, delta=deltas[i].clone().detach(), iterations=100).squeeze(1).cpu()
                
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
        print('Epoch : {}, Training Accuracy : {:.2f}%,  Test Accuracy : {:.2f}% \n'.format(
            epoch+1, train_acc[-1], test_acc[-1]))

        if augment:
            torch.save(deltas, f'augmented_deltas_epoch_{epoch}.pt')
        
        #if augmented_data:
        #    new_data = torch.cat(augmented_data, dim=0).detach().cpu()
        #    new_data = new_data.squeeze(1)
        #    new_labels = torch.cat(augmented_labels, dim=0).detach().cpu()
        #    train_loader.dataset.data = new_data
        #    train_loader.dataset.targets = new_labels
        #    torch.save(new_data, f'augmented_data_epoch_{epoch}.pt')
        #    torch.save(new_labels, f'augmented_labels_epoch_{epoch}.pt')
        
    return model, train_acc, test_acc












# def train(model, train_loader, test_loader, device, calc_sharpness, epochs):
#     model = model.to(device)
#     lr = 0.001
#     optimizer_SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     train_acc = []
#     test_acc = []

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         correct = 0
#         for X, Y, i in tqdm(train_loader):            
#             X = X.to(device)
#             Y = Y.to(device)
            
#             optimizer_SGD.zero_grad()
#             hypothesis = model(X)
#             loss = criterion(hypothesis, Y)
#             loss.backward()
#             optimizer_SGD.step()
#             predicted = torch.argmax(hypothesis, 1)
            
#             correct += (predicted == Y).sum().item()

#         train_acc.append(100. * correct / len(train_loader.dataset))
#         correct = 0
        
#         with torch.no_grad():
#             model.eval()
#             for X, Y, i in test_loader:
#                 X = X.to(device)
#                 Y = Y.to(device)
            
#                 hypothesis = model(X)
                
#                 predicted = torch.argmax(hypothesis, 1)
#                 correct += (predicted == Y).sum().item()
        
#         test_acc.append(100. * correct / len(test_loader.dataset))         
#         print('Epoch : {}, Training Accuracy : {:.2f}%,  Test Accuracy : {:.2f}% \n'.format(
#             epoch+1, train_acc[-1], test_acc[-1]))
    
#     hessian = None
#     if calc_sharpness:
#         hessian = calculate_model_hessian(model, criterion, test_loader)

#     return model, train_acc, test_acc, hessian

# def train_sam(model, train_loader, test_loader, device, calc_sharpness, epochs):
#     model = model.to(device)
#     base_optimizer = torch.optim.SGD
#     lr = 0.001
#     optimizer_SAM = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     train_acc = []
#     test_acc = []

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         correct = 0
        
#         for X, Y in train_loader:
#             X = X.to(device)
#             Y = Y.to(device)
            
#             optimizer_SAM.zero_grad()
#             hypothesis = model(X)
#             loss = criterion(hypothesis, Y)
#             loss.backward()
#             optimizer_SAM.first_step(zero_grad=True)
#             criterion(model(X), Y).backward()
#             optimizer_SAM.second_step(zero_grad=True)

#             predicted = torch.argmax(hypothesis, 1)
#             correct += (predicted == Y).sum().item()

#         train_acc.append(100. * correct / len(train_loader.dataset))
#         correct = 0
        
#         with torch.no_grad():
#             model.eval()
#             for X, Y in test_loader:
#                 X = X.to(device)
#                 Y = Y.to(device)
            
#                 hypothesis = model(X)
                
#                 predicted = torch.argmax(hypothesis, 1)
#                 correct += (predicted == Y).sum().item()
                
                
#         test_acc.append(100. * correct / len(test_loader.dataset))         
#         print('Epoch : {}, Training Accuracy : {:.2f}%,  Test Accuracy : {:.2f}% \n'.format(
#             epoch+1, train_acc[-1], test_acc[-1]))
    
#     hessian = None
#     if calc_sharpness:
#         hessian = calculate_model_hessian(model, criterion, test_loader)

#     return model, train_acc, test_acc, hessian

# def train_augment(model, train_loader, test_loader, device, calc_sharpness, epochs):
#     print("Starting augmented training")
#     model = model.to(device)
#     base_optimizer = torch.optim.SGD
#     lr = 0.001
#     optimizer_SAM = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     train_acc = []
#     test_acc = []

#     cosines = [[]]*len(list(model.parameters()))

#     deltas = (0.001**0.5)*torch.randn(train_loader.dataset.data.shape)


#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         correct = 0
#         augmented_data = []
#         augmented_labels = []   # Necessary if shuffling is enabled as indices need to be matched
        
#         for X, Y, i in train_loader:
#             X = X.to(device)
#             X.requires_grad_()
#             Y = Y.to(device)
#             optimizer_SAM.zero_grad()
#             hypothesis = model(X)
#             loss = criterion(hypothesis, Y)
#             loss.backward()
#             optimizer_SAM.first_step(zero_grad=True)
#             loss = criterion(model(X), Y)
#             loss.backward()
#             optimizer_SAM.second_step(zero_grad=False)
#             print("Creating augmented data")
#             deltas[i] = augment_data(X, Y, criterion, model, device, delta=deltas[i].clone().detach(), iterations=100).squeeze(1).cpu()
#             optimizer_SAM.zero_grad()
            
#             predicted = torch.argmax(hypothesis, 1)
#             correct += (predicted == Y).sum().item()

#         train_acc.append(100. * correct / len(train_loader.dataset))
#         correct = 0
        
#         with torch.no_grad():
#             model.eval()
#             for X, Y, i in test_loader:
#                 X = X.to(device)
#                 Y = Y.to(device)

#                 hypothesis = model(X)
                
#                 predicted = torch.argmax(hypothesis, 1)
#                 correct += (predicted == Y).sum().item()                
                
#         test_acc.append(100. * correct / len(test_loader.dataset))
#         print('Epoch : {}, Training Accuracy : {:.2f}%,  Test Accuracy : {:.2f}% \n'.format(
#             epoch+1, train_acc[-1], test_acc[-1]))

#         torch.save(deltas, f'augmented_deltas_epoch_{epoch}.pt')
        
#         #if augmented_data:
#         #    new_data = torch.cat(augmented_data, dim=0).detach().cpu()
#         #    new_data = new_data.squeeze(1)
#         #    new_labels = torch.cat(augmented_labels, dim=0).detach().cpu()
#         #    train_loader.dataset.data = new_data
#         #    train_loader.dataset.targets = new_labels
#         #    torch.save(new_data, f'augmented_data_epoch_{epoch}.pt')
#         #    torch.save(new_labels, f'augmented_labels_epoch_{epoch}.pt')
    
#     hessian = None
#     if calc_sharpness:
#         hessian = calculate_model_hessian(model, criterion, test_loader)
    
#     return model, train_acc, test_acc, hessian

