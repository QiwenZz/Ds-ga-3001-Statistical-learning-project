import matplotlib.pyplot as plt
from src.train_utils import train, evaluate, evaluate_se
import torch
import numpy as np
import os
from torch.optim.lr_scheduler import LambdaLR
import math
import copy
from src.utils import *
import json

def train_model(network, dataloaders, args, device):
    train_loader, val_loader = dataloaders
    epochs = args['epochs']

    criterion = torch.nn.CrossEntropyLoss()
    
    if args['optimizer'] == 'Adam':
        if args['model'] == 'deit':
            optimizer = torch.optim.Adam(network[1].parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    if args['optimizer'] == 'SGD':
        if args['model'] == 'deit':
            optimizer = torch.optim.SGD(network[1].parameters(), lr=args['lr'], weight_decay=args['weight_decay'],\
                                        momentum=args['momentum'])
        else:
            optimizer = torch.optim.SGD(network.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'])
    if args['scheduler'] == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2) #add more schedulers
    elif args['scheduler'] == 'cosineannealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0001)
    
    # main train loop
    best_acc = 0  # best val accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    early_stop = EarlyStopping(args['patience'])
    metrics = []
    for epoch in range(start_epoch, start_epoch+epochs):
        train_acc, train_loss = train(epoch, network, train_loader, criterion, optimizer, device, args)
        val_acc, val_loss = evaluate(epoch, network, val_loader, criterion, device, args)
        metrics.append([train_acc, train_loss, val_acc, val_loss])
        if args['scheduler'] == 'cosineannealing':
            scheduler.step(epoch)
        else:
            scheduler.step()
        
        early_stop(val_loss)
        if early_stop.early_stop:
            break

        # Save checkpoint.
        if val_acc > best_acc:
            print('Saving..')
            if args['model'] == 'deit':
                weight = network[1].state_dict()
            else:
                weight = network.state_dict() 
            state = {
                'net': weight,
                'acc': val_acc,
                'epoch': epoch,
                'args' : args
            }
            best_acc = val_acc

    state.update({'metrics': np.array(metrics)})
    
    torch.save(state,'./models/'+str(best_acc)+'.pth')
    
    if args['log']:
        # save args into json dictionary
        with open('./models/'+str(best_acc)+".txt", "w") as fp:
            json.dump(args, fp, indent=2)
    
    return best_acc
    
    
    
def _set_scheduler(optimizer, n_estimators, n_iters):
    """
    Set the learning rate scheduler for snapshot ensemble.
    Please refer to the equation (2) in original paper for details.
    """
    T_M = math.ceil(n_iters / n_estimators)
    lr_lambda = lambda iteration: 0.5 * (
        torch.cos(torch.tensor(math.pi * (iteration % T_M) / T_M)) + 1
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler
    
    
def train_model_se(network, dataloaders, args, device):
    train_loader, val_loader = dataloaders
    epochs = args['epochs']
    n_estimators = args['estimators']

    criterion = torch.nn.CrossEntropyLoss()
    if args['optimizer'] == 'Adam':
        if args['model'] == 'deit':
            optimizer = torch.optim.Adam(network[1].parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    if args['optimizer'] == 'SGD':
        if args['model'] == 'deit':
            optimizer = torch.optim.SGD(network[1].parameters(), lr=args['lr'], weight_decay=args['weight_decay'],\
                                        momentum=args['momentum'])
        else:
            optimizer = torch.optim.SGD(network.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'])
    
    scheduler = _set_scheduler(optimizer, n_estimators, epochs)

    # main train loop
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_acc = 0
    early_stop = EarlyStopping(args['patience'])
    metrics = []
    snapshots = []
    counter = 0  # a counter on generating snapshots
    total_iters = 0
    n_iters_per_estimator = epochs * len(train_loader) // n_estimators
    for epoch in range(start_epoch, start_epoch+epochs):

        
        if args['model'] == 'deit':
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0
            teacher_model, student_model = network
            teacher_model.train()
            student_model.train()

            for batch_idx, (inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(device), targets.to(device)

                student_outputs = student_model(inputs)
                _, student_predictions = torch.max(student_outputs.data, 1)

                if not args['student_only']:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    loss = 0.5*criterion(student_outputs, targets) + 0.5*criterion(teacher_outputs, targets)
                else:
                    loss = criterion(student_outputs, targets)

                train_loss += loss.item()
                total += targets.size(0)
                correct += torch.sum(student_predictions == targets).item()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



                progress_bar(batch_idx, len(train_loader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
        else:
            # Ensemble training
            print('\nEpoch: %d' % epoch)
            network.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)


                outputs = network(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                scheduler.step()
                counter += 1
                total_iters += 1



                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_acc, train_loss = correct/total, train_loss/(batch_idx+1)

        # Save snapshot
        if counter % n_iters_per_estimator == 0:
            if args['model'] == 'deit':
                snapshot = copy.deepcopy(network[1])
                snapshots.append(snapshot.state_dict())
            else:
                snapshot = copy.deepcopy(network)
                snapshots.append(snapshot.state_dict())

        # Ensemble evaluation, combine the predictions on the evaluation data
        # of the current model, and the current set of snapshots
        #if args['model'] == 'deit':
        #    snapshots = snapshots+[copy.deepcopy(network[1]).state_dict()]
        #else:
        #    snapshots = snapshots+[copy.deepcopy(network).state_dict()]
        val_acc, val_loss =  evaluate_se(epoch, network, snapshots+[copy.deepcopy(network).state_dict() if not args['model'] == 'deit' else copy.deepcopy(network[1]).state_dict()],\
                                val_loader, criterion, device, args, method=args['voting'],verbose=True)
        ######
        best_acc = max(best_acc, val_acc)
        metrics.append([train_acc, train_loss, val_acc, val_loss])
        
        early_stop(val_loss)
        if early_stop.early_stop:
            break

    state = {'snapshots': snapshots, 'metrics': np.array(metrics), 'args':args}
    torch.save(state,  './models/'+str(best_acc)+'.pth')
    if args['log']:
        # save args into json dictionary
        with open('./models/'+str(best_acc)+".txt", "w") as fp:
            json.dump(args, fp, indent=2)
    
    return best_acc