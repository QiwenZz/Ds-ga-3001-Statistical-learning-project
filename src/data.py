from imblearn.over_sampling import SMOTE
from torchvision import datasets, transforms, models
import os
import pandas as pd
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

def get_data(fulldir):
    classes = os.listdir(fulldir)
    classes.remove('.DS_Store')
    classes.sort()
    class_to_idx = dict(zip(classes, range(len(classes)))) 
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    resizer =  transforms.Resize((324,324))
    convert_tensor = transforms.ToTensor()
    X = []
    y = []
    print('retrieving data')
    for i, label in idx_to_class.items():
        path = fulldir+"/"+label
        for file in tqdm(os.listdir(path)):
            X.append(torch.unsqueeze(convert_tensor(resizer(Image.open(path+'/'+file).convert('RGB'))),0))
            y.append(i)
    
    return torch.cat(X,dim=0),y, class_to_idx, idx_to_class

class PlantDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        item = self.X[idx]
        if self.transform:
            item = self.transform(item)
        return item, self.y[idx]
    
def smote_balance(X_train,y_train):
    X_resampled, y_resampled = SMOTE(n_jobs=-1).fit_resample(X_train.reshape(len(X_train),-1), y_train)
    return torch.from_numpy(X_resampled.reshape((len(X_resampled),3,324,324))), y_resampled

class RandomAddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05,prob=0.5):
        self.std = std
        self.mean = mean
        self.prob = prob
        
    def __call__(self, tensor):
        if np.random.choice(a=[0,1], p=[1-self.prob,self.prob]):
            return tensor + torch.randn(tensor.shape) * self.std + self.mean
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def get_dataloaders(path, args):
    
    # read in data
    X,y, class_to_idx, idx_to_class = get_data(path)
    
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,stratify=y)
    
    # whether using smote
    if args['smote']:
        X_resampled, y_resampled = smote_balance(X_train,y_train)
    else:
        X_resampled, y_resampled = X_train, y_train
        
        
    # set up for transformation and augmentation
    train_transform = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomAddGaussianNoise(std=args['noise_std']),
    transforms.ColorJitter(brightness=args['brightness']),
    transforms.Normalize(mean=args['normalization_mean'], std=args['normalization_std'])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=args['normalization_mean'], std = args['normalization_std'])
    ])
    
    train_loader = DataLoader(train_dataset, batch_size = args['bz'],shuffle=args['shuffle'], drop_last=True )
    val_loader = DataLoader(val_dataset, batch_size = args['bz'], drop_last=True )
    
    return train_loader, val_loader
    