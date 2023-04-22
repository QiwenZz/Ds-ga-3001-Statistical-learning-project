#!pip install gans-implementations
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
import pickle
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

#!pip install gans-implementations
from gans_package.models import GAN_Generator, GAN_Discriminator

def get_data(fulldir, args):
    classes = os.listdir(fulldir)
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')
    classes.sort()
    class_to_idx = dict(zip(classes, range(len(classes)))) 
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    resizer =  transforms.Resize(args['size'])
    convert_tensor = transforms.ToTensor()
    X = []
    y = []
    print('transforming data')
    for i, label in idx_to_class.items():
        path = fulldir+"/"+label
        for file in tqdm(os.listdir(path)):
            X.append(torch.unsqueeze(convert_tensor(resizer(Image.open(path+'/'+file).convert('RGB'))),0))
            y.append(i)
    
    os.makedirs(f"data/processed_{args['size']}")
    
    # save X
    torch.save(torch.cat(X,dim=0), f"data/processed_{args['size']}/X.pt")
    
    # save y
    with open(f"data/processed_{args['size']}/y.pickle", "wb") as fp:   
        pickle.dump(y, fp)
    
    # save idx_to_class dictionary
    with open(f"data/processed_{args['size']}/idx_to_class.txt", "w") as fp:
        json.dump(idx_to_class, fp)
    

def import_processed_data(args):
    X = torch.load(f"data/processed_{args['size']}/X.pt")
    
    with open(f"data/processed_{args['size']}/y.pickle", "rb") as fp:
        y = pickle.load(fp)
        
    with open(f"data/processed_{args['size']}/idx_to_class.txt", "r") as fp:
        idx_to_class = json.load(fp)        

    return X, y, idx_to_class


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

def get_dataloaders(path, args):

    if not os.path.isdir(f"data/processed_{args['size']}"):
        # transform data
        get_data(path, args)
    
    X, y, idx_to_class = import_processed_data(args)
    
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,stratify=y)
    

    X_resampled, y_resampled = X_train, y_train
        
        
    # set up for transformation and augmentation
    
    #transform to torch dataset object
    train_dataset = PlantDataset(X_resampled,y_resampled)
    val_dataset = PlantDataset(X_val,y_val)
    
    #construct dataloader
    train_loader = DataLoader(train_dataset, batch_size = args['bz'],shuffle=args['shuffle'], drop_last=True )
    val_loader = DataLoader(val_dataset, batch_size = args['bz'], drop_last=True )
    
    return train_loader, val_loader, idx_to_class
    
def train_gan(train_loader): 
    loss_hist = {}
    loss_hist["generator loss"] = []
    loss_hist["discriminator loss"] = []
    for epoch in range(1, NUM_EPOCHES+1):
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        for real, label in train_loader:
            b = real.size(0)
            real = real.view(b, -1).to(device)
            
            d_optimizer.zero_grad()
            noise = torch.randn(b, latent_dim).to(device)
            fake = g(noise)
            fake_pred = d(fake.detach())
            real_pred = d(real)
            fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
            real_loss = criterion(real_pred, torch.ones_like(real_pred))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            epoch_d_loss += d_loss.item()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            noise = torch.randn(b, latent_dim).to(device)
            fake = g(noise)
            pred = d(fake)
            g_loss = criterion(pred, torch.ones_like(pred))
            g_loss.backward()
            epoch_g_loss += g_loss.item()
            g_optimizer.step()
        
        epoch_g_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)
        
        loss_hist["generator loss"].append(epoch_g_loss)
        loss_hist["discriminator loss"].append(epoch_d_loss)
        
        print("Epoch {}: Generator Loss: {}; Discriminator Loss: {}".format(epoch, epoch_g_loss, epoch_d_loss))
        
        if epoch%5==0:
            noise = torch.randn(b, latent_dim).to(device)
            fake = g(noise)
        
            images = fake.detach().to("cpu").view(-1, 3, args['vanilla_size'], args['vanilla_size'])
            images_grid = make_grid(images[:9], nrow=3)
            plt.imshow(images_grid.permute(1, 2, 0).squeeze())
            plt.show()
    torch.save(g.state_dict(), 'vanilla_g.pt')
    torch.save(d.state_dict(), 'vanilla_d.pt')
    return loss_hist

def predict_gan(val_loader, args, idx_to_class): 
    # f"../vanilla/{key}/{count}.png"
    if not os.path.isdir("/data/vanilla"):
        print('here')
        for key in list(idx_to_class.values()): 
            os.makedirs(f"/data/vanilla/{key}")
    #g = GAN_Generator(latent_dim, g_out_size, g_hidden_size, g_num_layers).to(device)
    #d = GAN_Discriminator(g_out_size, d_hidden_size, d_num_layers).to(device)
    #g.load_state_dict(torch.load('vanilla_g.pt'))
    #d.load_state_dict(torch.load('vanilla_d.pt'))
    with torch.no_grad(): 
        count = 0
        for real, label in val_loader:
            b = real.size(0)
            real = real.view(b, -1).to(device)


            noise = torch.randn(b, latent_dim).to(device)
            fake = g(noise)
        
            images = fake.detach().to("cpu").view(-1, 3, args['vanilla_size'], args['vanilla_size'])
            for i in range(images.shape[0]): 
                image = images[i]
                lab = label[i]
                #print(str(lab.numpy()))
                classes = idx_to_class[str(lab.numpy())]
                save_image(image, f"/data/vanilla/{classes}/{count}.png")
                count += 1

def main(args, path): 
    BATCH_SIZE = 64
    LR = 1e-5
    NUM_EPOCHES = 200
    latent_dim = 64
    g_out_size = 3 * args['vanilla_size'] * args['vanilla_size']
    g_hidden_size = 128
    g_num_layers = 4

    d_hidden_size = 512
    d_num_layers = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    g = GAN_Generator(latent_dim, g_out_size, g_hidden_size, g_num_layers).to(device)
    d = GAN_Discriminator(g_out_size, d_hidden_size, d_num_layers).to(device)

    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(g.parameters(), lr=LR)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=LR)

    train_loader , val_loader, idx_to_class = get_dataloaders(path)    
    predict_gan(val_loader, args, idx_to_class)