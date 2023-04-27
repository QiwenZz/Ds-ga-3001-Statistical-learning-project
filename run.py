import torch
import argparse
import os, sys, json
from src.data import get_dataloaders
from src.engine import train_model, train_model_se
from src.model import load_model
from src.out import write_out_submission
from src.utils import tuple_float_type,tuple_int_type
import optuna

parser = argparse.ArgumentParser()

# Data Tuning Related
parser.add_argument('--path', default='data/train', type=str,
                    help='path of the root data foler')
parser.add_argument('--smote', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether using smote for data augmentation')
parser.add_argument('--smote_k', default=5, type=int,
                    help='The nearest neighbors used to define the neighborhood of samples to use to generate the synthetic samples')
parser.add_argument('--size', default="(399,399)", type=tuple_int_type,
                    help='size to resize')
parser.add_argument('--bz', default=64, type=int,
                    help='batch size')
parser.add_argument('--norm_mean', default="(0.3272, 0.2874, 0.2038)", type=tuple_float_type,
                    help='Mean value of z-scoring normalization for each channel in image')
parser.add_argument('--norm_std', default="(0.0965, 0.1009, 0.1173)", type=tuple_float_type,
                    help='Mean value of z-scoring standard deviation for each channel in image')
parser.add_argument('--brightness', default="(0.8,2)", type=tuple_float_type,
                    help='Brightness range for data augmentation')
parser.add_argument('--noise_std', default=0.05, type=float,
                    help='Adding noise with guassian distribution for data augmentation')
parser.add_argument('--shuffle', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to shuffle training data during optimization')

# Hardware Related
parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')   

# Training Arguments
parser.add_argument('--model', default='resnet50', type=str,
                    help='the model to use for training or make predictions')
parser.add_argument('--log', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to generate a json file with args along with the model')
parser.add_argument('--reuse_model', default='', type=str,
                    help='the name of the model to reuse for finetuning on new data')
parser.add_argument('--optuna', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether use optuna to find the best hyperparameters')
parser.add_argument('--optuna_trials', default=3, type=int,
                    help='number of trials to find the best parameter, the larger the better')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='the optimizer to use')
parser.add_argument('--scheduler', default='cosineannealing', type=str,
                    help='what scheduler to use')
parser.add_argument('--lr', default=0.001, type=float,
                    help='the optimizers learning rate')  
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum factor for sgd')  
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help=' weight decay (L2 penalty) for both sgd and adam')  
parser.add_argument('--freeze_num', default=7, type=int,
                    help='number of layers to freeze during fine tuning')  
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs')   
parser.add_argument('--patience', default=5, type=int,
                    help='patience for early stop')   

# Ensembling Arguments
parser.add_argument('--snapshot_ensemble', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether snapshot ensembling to be performed')
parser.add_argument('--estimators', default=10, type=int,
                    help='number of estimators to be ensembled')  
parser.add_argument('--voting', default='majority', type=str,
                    help='how the ensembling between different estimators is performed')  

# Deit related
parser.add_argument('--teacher', default='deit_base_distilled_patch16_224', type=str,
                    help='teacher model for deit')
parser.add_argument('--student', default='deit_small_distilled_patch16_224', type=str,
                    help='student model for deit')

# Testing Model Related
parser.add_argument('--test', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether test a model and make prediction file')
parser.add_argument('--test_path', default='data/test', type=str,
                    help='path of the test data foler')
parser.add_argument('--test_model', default='0.9787946428571429.pth', type=str,
                    help='the model that will be used to produce predictions')

args = vars(parser.parse_args())


def objective(trial,args,device,dataloaders):
    # Define the hyperparameters to be tuned
    lr = trial.suggest_float(
        "lr", 1e-6, 1e-2, log=True
    )
    momentum = trial.suggest_float("momentum", 0.0, 1.0)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    
    # Pass the hyperparameters to the load_model function
    args['lr'] = lr
    args['momentum'] = momentum
    args['weight_decay'] = weight_decay
    
    if args['model'] == 'deit':
        network = load_model(args['model'],args,device)
    else:
        network = load_model(args['model'],args,device)
    if args['snapshot_ensemble']:
        best_acc = train_model_se(network, dataloaders, args, device)
    else:
        best_acc = train_model(network, dataloaders, args, device)

    return best_acc


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
    if args['test']:
        write_out_submission(args, device)
    else:
        dataloaders = get_dataloaders(args['path'], args)
        if args['optuna']:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, args,device,dataloaders), n_trials=args['optuna_trials'])
            # Print the best hyperparameters and validation accuracy
            print('Best hyperparameters: ', study.best_params)
            print('Best validation accuracy: ', study.best_value)
            with open('./models/'+str(study.best_value)+"best_params.txt", "w") as fp:
                json.dump(study.best_params, fp, indent=2)
        else:
            if args['model'] == 'deit':
                network = load_model(args['model'],args,device)
            else:
                network = load_model(args['model'],args,device)
            if args['snapshot_ensemble']:
                train_model_se(network, dataloaders, args, device)
            else:
                train_model(network, dataloaders, args, device)
        
    
    
if __name__ == '__main__':
    main(args)