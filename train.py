import torch
import torch.optim
from torch import nn
from poison_data import poison_data
from hessian import calculate_spectrum
from tqdm import tqdm
from sam import SAM
import optuna

def train(args):
    if args['train_normal']:
        print("Starting training")
    elif args['sam']:
        print("Starting training with SAM")
    elif args['poison']:
        print("Starting training with poisoning")
    
    model = args['model'].to(args['device'])
    train_acc = []
    val_acc = []
    test_acc = []
    criterion = nn.CrossEntropyLoss()
    early_stopping_epochs = 10

    if args['sam'] or args['poison']:
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=args['lr'], momentum=args['momentum'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    
    if args['poison']:
        deltas = (0.001**0.5)*torch.randn(args['train_loader'].dataset.data.shape)

    for epoch in range(args['epochs']):
        model.train()
        correct = 0
        
        for X, Y, i in iter(args['train_loader']):
            X = X.to(args['device'])
            X.requires_grad_()
            Y = Y.to(args['device'])

            transformed_X = args['augmentation'](X)    # In case no augmentation is needed, this is the identity function

            optimizer.zero_grad()
            hypothesis = model(transformed_X)
            loss = criterion(hypothesis, Y)
            loss.backward(retain_graph=True)
            
            if args['sam'] or args['poison']:
                optimizer.first_step(zero_grad=True)
                loss = criterion(model(transformed_X), Y)
                loss.backward()
                optimizer.second_step(zero_grad=False)

                if args['poison'] and epoch >= args['poison_start_epoch']:
                    deltas[i] = poison_data(X, Y, criterion, model, args['device'], delta=deltas[i].clone().detach(), iterations=args['iterations'], epsilon=args['epsilon'], lr=args['poison_lr']).squeeze(1).detach().cpu()
                
                optimizer.zero_grad()
            else:
                optimizer.step()
            
            predicted = torch.argmax(hypothesis, 1)
            correct += (predicted == Y).sum().item()

            del X, Y, transformed_X, hypothesis, loss, predicted
            torch.cuda.empty_cache()

        train_acc.append(100. * correct / len(args['train_loader'].dataset))
        correct = 0
        
        with torch.no_grad():
            model.eval()

            for X, Y, i in args['test_loader']:
                X = X.to(args['device'])
                Y = Y.to(args['device'])

                hypothesis = model(X)
                predicted = torch.argmax(hypothesis, 1)
                correct += (predicted == Y).sum().item()                
        
        test_acc.append(100. * correct / len(args['test_loader'].dataset))
        correct = 0
        
        with torch.no_grad():
            model.eval()

            for X, Y, i in args['val_loader']:
                X = X.to(args['device'])
                Y = Y.to(args['device'])

                hypothesis = model(X)
                predicted = torch.argmax(hypothesis, 1)
                correct += (predicted == Y).sum().item()                
        
        cur_val_acc = 100. * correct / len(args['val_loader'].dataset)
        val_acc.append(cur_val_acc)
        print(f'Epoch: {epoch+1}, Training Accuracy: {train_acc[-1]}, \
              Validation Accuracy: {val_acc[-1]}, Test Accuracy: {test_acc[-1]}, Seed: {args['seed']}')
        
        if 'trial' not in args:
            args['metrics_logger'].log_epoch_acc(epoch, train_acc[-1], val_acc[-1], test_acc[-1])
            args['metrics_logger'].save_model(model)

        if args['poison'] and epoch > args['poison_start_epoch'] and 'trial' not in args and cur_val_acc == max(val_acc):
            args['metrics_logger'].save_deltas(deltas)
        
        if 'trial' in args and not args['poison'] and args["should_prune"]:
            args['trial'].report(val_acc[-1], step=epoch)
            if args['trial'].should_prune():
                raise optuna.TrialPruned()
        
        if args['early_stopping'] and len(val_acc) >= early_stopping_epochs:
            if max(val_acc) not in val_acc[-early_stopping_epochs:]:
                print(f"Stopping early after {epoch} epochs")
                break
        
        torch.cuda.empty_cache()
    
    if args['poison']:
        args['metrics_logger'].save_final_deltas(deltas)
    
    if args['calculate_sharpness']:
        print(calculate_spectrum(model, args['test_loader'], criterion, 20))
    
    return model, train_acc, val_acc, test_acc