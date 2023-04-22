from vanilla_process import *
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
#from vanilla_process import *

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def image_segmentation(fulldir): 
    for class_folder_name in os.listdir(fulldir):
        class_folder_path = os.path.join(fulldir, class_folder_name)
        os.makedirs(f"data/segmentation/{class_folder_name}")
        for image_name in tqdm(os.listdir(class_folder_path)):
            image_org = cv2.imread(os.path.join(class_folder_path, image_name), cv2.IMREAD_COLOR)
            image_mask = create_mask_for_plant(image_org)
            image_segmented = segment_plant(image_org)
            image_sharpen = sharpen_image(image_segmented)
            cv2.imwrite(os.path.join('data/segmentation', class_folder_name, image_name), image_sharpen)

def get_data(fulldir, args, seg=''):
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
    
    os.makedirs(f"data/processed_{args['size']}{seg}")
    
    # save X
    torch.save(torch.cat(X,dim=0), f"data/processed_{args['size']}{seg}/X.pt")
    
    # save y
    with open(f"data/processed_{args['size']}{seg}/y.pickle", "wb") as fp:   
        pickle.dump(y, fp)
    
    # save idx_to_class dictionary
    with open(f"data/processed_{args['size']}{seg}/idx_to_class.txt", "w") as fp:
        json.dump(idx_to_class, fp)
    

def import_processed_data(args, path):
    X = torch.load(f"{path}/X.pt")
    
    with open(f"{path}/y.pickle", "rb") as fp:
        y = pickle.load(fp)
        
    with open(f"{path}/idx_to_class.txt", "r") as fp:
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
    
def smote_balance(X_train,y_train,args):
    X_resampled, y_resampled = SMOTE(n_jobs=-1).fit_resample(X_train.reshape(len(X_train),-1), y_train)
    return torch.from_numpy(X_resampled.reshape((len(X_resampled),3,args['size'][0],args['size'][1]))), y_resampled

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

    if not os.path.isdir(f"data/processed_{args['size']}"):
        # transform data
        get_data(path, args)

    if args['segmentation']: 
        if not os.path.isdir(f"data/segmentation"):
            image_segmentation(path)
        if not os.path.isdir(f"data/processed_{args['size']}_seg"):
            get_data('data/segmentation', args, '_seg')
    elif args['vanilla']: 
        vanilla_process.main_vanilla()
        get_data('data/vanilla', args)

    if args['segmentation']: 
        X_original, y_original, idx_to_class = import_processed_data(args, f"data/processed_{args['size']}")
        X_new, y_new, idx_to_class = import_processed_data(args, f"data/processed_{args['size']}_seg")
        X = torch.cat((X_original,X_new), 0)
        y = y_original + y_new
    else: 
        X, y, idx_to_class = import_processed_data(args, f"data/processed_{args['size']}")

    
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,stratify=y)
    
    # whether using smote
    if args['smote']:
        X_resampled, y_resampled = smote_balance(X_train,y_train,args)
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
    transforms.Normalize(mean=args['norm_mean'], std=args['norm_std'])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=args['norm_mean'], std = args['norm_std'])
    ])
    
    #transform to torch dataset object
    train_dataset = PlantDataset(X_resampled,y_resampled,train_transform)
    val_dataset = PlantDataset(X_val,y_val,val_transform)
    
    #construct dataloader
    train_loader = DataLoader(train_dataset, batch_size = args['bz'],shuffle=args['shuffle'], drop_last=True )
    val_loader = DataLoader(val_dataset, batch_size = args['bz'], drop_last=True )
    
    return train_loader, val_loader