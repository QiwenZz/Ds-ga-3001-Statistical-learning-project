import matplotlib.pyplot as plt
from src.train_utils import train, evaluate
import torch
import numpy as np
import os

def train_model(network, dataloaders, args, device):
    train_loader, val_loader = dataloaders
    #epochs = args['epoch']
    epochs = 10

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)


    # main train loop
    best_acc = 0  # best val accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    metrics = []
    for epoch in range(start_epoch, start_epoch+epochs):
        train_acc, train_loss = train(epoch, network, train_loader, criterion, optimizer, device)
        val_acc, val_loss = evaluate(epoch, network, val_loader, criterion, device)
        metrics.append([train_acc, train_loss, val_acc, val_loss])
        scheduler.step()

        # Save checkpoint.
        if val_acc > best_acc:
            print('Saving..')
            state = {
                'net': network.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            current_directory = os.getcwd()
            print(current_directory)
            final_directory = os.path.join(current_directory, r'models')
            if not os.path.isdir(final_directory):
                os.mkdir(final_directory)
            torch.save(state, './models/'+' '.join(args)+'.pth')
            best_acc = val_acc

    state = torch.load('./models/'+' '.join(args)+'.pth')
    state.update({'metrics': np.array(metrics)})
    torch.save(state, './models/'+' '.join(args)+'.pth')
    
    
    
    
    
    