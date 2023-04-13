from src.utils import progress_bar
import torch


def train(epoch, network, trainloader, criterion, optimizer, device):

    print('\nEpoch: %d' % epoch)
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

        

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return correct/total, train_loss/(batch_idx+1)


def evaluate(epoch, network, valloader, criterion, device):
    network.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (eval_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return correct/total, eval_loss/(batch_idx+1)