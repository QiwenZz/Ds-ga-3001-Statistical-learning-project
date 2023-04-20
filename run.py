import torch
import argparse
import os, sys, json
from src.data import get_dataloaders
from src.engine import train_model, train_model_se
from src.model import load_model
from src.out import write_out_submission
from src.utils import tuple_float_type,tuple_int_type

parser = argparse.ArgumentParser()


# Data Loading Related
parser.add_argument('--X_path', default='data/processed/X.pt', type=str,
                    help='path of X tensor')
parser.add_argument('--y_path', default='data/processed/y.pickle', type=str,
                    help='path of y list')
parser.add_argument('--idx_to_class_path', default='data/processed/idx_to_class.txt', type=str,
                    help='path of the idx_to_class dictionary')

# Data Tuning Related
parser.add_argument('--path', default='data/train', type=str,
                    help='path of the root data foler')
parser.add_argument('--smote', default=True, type=bool,
                    help='whether using smote for data augmentation')
parser.add_argument('--size', default="(324,324)", type=tuple_int_type,
                    help='size to resize')
parser.add_argument('--bz', default=64, type=int,
                    help='batch size')
parser.add_argument('--norm_mean', default="(0.485,0.456,0.406)", type=tuple_float_type,
                    help='Mean value of z-scoring normalization for each channel in image')
parser.add_argument('--norm_std', default="(0.229,0.224,0.225)", type=tuple_float_type,
                    help='Mean value of z-scoring standard deviation for each channel in image')
parser.add_argument('--brightness', default="(0.8,2)", type=tuple_float_type,
                    help='Brightness range for data augmentation')
parser.add_argument('--noise_std', default=0.05, type=float,
                    help='Adding noise with guassian distribution for data augmentation')
parser.add_argument('--shuffle', default=True, type=bool,
                    help='whether to shuffle training data during optimization')

# Hardware Related
parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')   

# Training Arguments
parser.add_argument('--model', default='resnet50', type=str,
                    help='the model to use for training or make predictions')
parser.add_argument('--optimizer', default='SGD', type=str,
                    help='the optimizer to use')
parser.add_argument('--lr', default=0.1, type=float,
                    help='the optimizers learning rate')  
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs')   
parser.add_argument('--patience', default=5, type=int,
                    help='patience for early stop')   

# Ensembling Arguments
parser.add_argument('--snapshot_ensemble', default=False, type=bool,
                    help='whether snapshot ensembling to be performed')
parser.add_argument('--estimators', default=10, type=int,
                    help='number of estimators to be ensembled')  

# Testing Model Related
parser.add_argument('--test', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether test a model and make prediction file')
parser.add_argument('--test_path', default='data/test', type=str,
                    help='path of the test data foler')
parser.add_argument('--test_model', default='0.9564732142857143.pth', type=str,
                    help='the model that will be used to produce predictions')

args = vars(parser.parse_args())

def main(args):
    print(args['size'])
    print(type(args['size']))
    print(f'CUDA availability: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'GPU name: {torch.cuda.get_device_name(i)}')

    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("using cuda:{}".format(args['device_id']))
    else:
         print("using {}".format(device))
    if args['test']:
        write_out_submission(args, device)
    else:
        dataloaders = get_dataloaders(args['path'], args)
        network = load_model(model_name = args['model'], freeze_counter=7).to(device)
        if args['snapshot_ensemble']:
            train_model_se(network, dataloaders, args, device)
        else:
            train_model(network, dataloaders, args, device)
        
    
    
if __name__ == '__main__':
    main(args)