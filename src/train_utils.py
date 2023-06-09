from src.utils import progress_bar
import torch
import copy


def train(epoch, network, trainloader, criterion, optimizer, device, args):
    if args['model'] == 'deit':
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        teacher_model, student_model = network
        teacher_model.train()
        student_model.train()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
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

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.sum(student_predictions == targets).item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    else:

        print('\nEpoch: %d' % epoch)
        network.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if args['model'] == 'deit':
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

            outputs = network(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return correct/total, train_loss/(batch_idx+1)


def evaluate(epoch, network, valloader, criterion, device,args, verbose=True):
    if args['model'] == 'deit':
        _, network = network
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
            if verbose:
                progress_bar(batch_idx, len(valloader), 'Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
                            % (eval_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return correct/total, eval_loss/(batch_idx+1)


def evaluate_se(epoch, network, snapshots, valloader, criterion, device, args, method='average', verbose=True):
    model_list = [copy.deepcopy(network) for _ in range(len(snapshots))]
    if args['model'] == 'deit':
        model_list = [copy.deepcopy(network[1]) for _ in range(len(snapshots))]
    for model, weight in zip(model_list, snapshots):
        model.load_state_dict(weight)
        model.eval()

    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_list = [net(inputs) for net in model_list]
            outputs = torch.mean(torch.stack(outputs_list), 0).squeeze()
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            if method == 'average':
                _, predicted = outputs.max(1)
            elif method == 'majority':
                predicted = [preds.max(1)[1] for preds in outputs_list]
                predicted = torch.stack(predicted).mode(0).values

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if verbose:
                progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (eval_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return correct/total, eval_loss/(batch_idx+1)