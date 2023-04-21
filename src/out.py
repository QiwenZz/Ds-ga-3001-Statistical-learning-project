import os
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset, DataLoader
import numpy as np
import json
from src.model import load_model
import copy

class PlantTestDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        item = self.X[idx]
        return item

def test(test_loader, model, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for image in test_loader:
            image = image.to(device)
            y_pred = model(image)
            y_pred_labels = y_pred.argmax(dim=1)
            y_pred_labels = y_pred_labels.cpu().numpy()
            predictions.append(y_pred_labels)
    return predictions

def test_se(test_loader, network, snapshots, method, device):
    model_list = [copy.deepcopy(network) for _ in range(len(snapshots))]

    for model, weight in zip(model_list, snapshots):
        model.load_state_dict(weight)
        model.to(device)
        model.eval()
    
    predictions = []
    with torch.no_grad():
        for image in test_loader:
            image = image.to(device)
            pred_list = [net(image) for net in model_list]
            if method == 'average':
                _, y_pred_labels = (torch.mean(torch.stack(pred_list), 0).squeeze()).max(1)
            elif method == 'majority':
                y_pred_labels = [preds.max(1)[1] for preds in pred_list]
                y_pred_labels = torch.stack(y_pred_labels).mode(0).values
            
            y_pred_labels = y_pred_labels.cpu().numpy()
            predictions.append(y_pred_labels)
    return predictions
    
def write_out_submission(args,device):
    # load in past state
    state = torch.load('models/'+args['test_model'])
    
    # load in related transformation hyperparameter in the past state
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(state['args']['size']),
    transforms.Normalize(mean=state['args']['norm_mean'], std = state['args']['norm_std'])
    ])
    
    test_folder_path = args['test_path']
    X = []
    for file in os.listdir(test_folder_path):
        X.append(torch.unsqueeze(test_transform(Image.open(test_folder_path+'/'+file).convert('RGB')),0))
    
    test_dataset = PlantTestDataset(torch.cat(X,dim=0))
    test_loader = DataLoader(test_dataset, batch_size = args['bz'],shuffle=False)

    # load in state and model
    model = load_model(state['args']['model'],args)

    if state['args']['snapshot_ensemble']:
        snapshots = state['snapshots']
        preds = test_se(test_loader, model, snapshots, args['voting'], device)
    else:
        model.load_state_dict(state['net'])
        preds = test(test_loader, model, device)
    
    
    # make predictions
    predictions = np.concatenate(preds).ravel()
    
    
    file = pd.DataFrame(os.listdir(test_folder_path), columns = ['file'])
    file['species'] = predictions
    file['species'] = file['species'].astype('str')
    
    with open(f"data/processed_{args['size']}/idx_to_class.txt", "r") as fp:
        idx_to_class = json.load(fp)         
        
    file = file.replace({"species": idx_to_class})
    
    # change the name after having json file 
    if os.path.exists("result"):
        file.to_csv("result/"+args['test_model'][:5]+".csv", index=False)
    else:
        os.makedirs("result")
        file.to_csv("result/"+args['test_model'][:5]+".csv", index=False)

    
    