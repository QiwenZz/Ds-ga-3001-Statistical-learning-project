import torch
import argparse
import os, sys, json
from src.data import get_dataloaders


parser = argparse.ArgumentParser()


# Data Related
parser.add_argument('--path', default='data/train', type=str,
                    help='path of the root data foler')
parser.add_argument('--smote', default=True, type=bool,
                    help='whether using smote for data augmentation')
parser.add_argument('--bz', default=32, type=int,
                    help='batch size')
parser.add_argument('--normalization_mean', default=(0.3272, 0.2874, 0.2038), type=tuple,
                    help='Mean value of z-scoring normalization for each channel in image')
parser.add_argument('--normalization_std', default=(0.0965, 0.1009, 0.1173), type=tuple,
                    help='Mean value of z-scoring standard deviation for each channel in image')
parser.add_argument('--brightness', default=(0.8,2), type=tuple,
                    help='Brightness range for data augmentation')
parser.add_argument('--noise_std', default=0.05, type=float,
                    help='Adding noise with guassian distribution for data augmentation')

# Hardware Related
parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')   

args = vars(parser.parse_args())

def main(args):

    print(f'CUDA availability: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'GPU name: {torch.cuda.get_device_name(i)}')

    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("using cuda:{}".format(args['device_id']))
    else:
         print("using {}".format(device))
    dataloaders = get_dataloaders(args['path'], args)

    return dataloaders
if __name__ == '__main__':
    main(args)