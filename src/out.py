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
    predictions = []
    with torch.no_grad():
        for image in test_loader:
            image = image.to(device)
            y_pred = model(image)
            y_pred_labels = y_pred.argmax(dim=1)
            y_pred_labels = y_pred_labels.cpu().numpy()
            predictions.append(y_pred_labels)
    return predictions

def write_out_submission(args,device):
    
    # load in model
    model = load_model(model_name = args['model'])
    model.load_state_dict(torch.load('models/'+args['test_model'])['net'])
    model.eval()
    
    # load in test data
    # change later when having config file ready for model saved
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((324,324)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    ])
    
    test_folder_path = args['test_path']
    X = []
    for file in os.listdir(test_folder_path):
        X.append(torch.unsqueeze(test_transform(Image.open(test_folder_path+'/'+file).convert('RGB')),0))
    
    test_dataset = PlantTestDataset(torch.cat(X,dim=0))
    test_loader = DataLoader(test_dataset, batch_size = args['bz'],shuffle=False)
    
    
    # make predictions
    predictions = np.concatenate(test(test_loader, model, device)).ravel()
    
    
    file = pd.DataFrame(os.listdir(test_folder_path), columns = ['file'])
    file['species'] = predictions
    file['species'] = file['species'].astype('str')
    
    with open(args['idx_to_class_path'], "r") as fp:
        idx_to_class = json.load(fp)     
        
    
    file = file.replace({"species": idx_to_class})
    
    # change the name after having json file 
    if os.path.exists("result"):
        file.to_csv("result/1.csv", index=False)
    else:
        os.makedirs("result")
        file.to_csv("result/1.csv", index=False)

    
    