#!/bin/bash

python run.py --path data/train \
--smote True \
--smote_k 5 \
--size "(384,384)" \
--bz 32 \
--norm_mean "(0.3272, 0.2874, 0.2038)" \
--norm_std "(0.0965, 0.1009, 0.1173)" \
--brightness "(0.8,2)" \
--noise_std 0.05 \
--shuffle True \
--device_id 0 \
--model deit \
--reuse_model '' \
--optuna False \
--optuna_trials 5 \
--optimizer SGD \
--schedular cosineannealing \
--lr 0.001 \
--momentum 0.9 \
--weight_decay 1e-4 \
--freeze_num 7 \
--epochs 100 \
--patience 5 \
--snapshot_ensemble True \
--log True \
--estimators 10 \
--voting majority \
--teacher deit_base_distilled_patch16_384 \
--student deit_base_distilled_patch16_384 \
--test False \
--test_path data/test \
--test_model 0.9787946428571429.pth \

# # Data Tuning Related
# parser.add_argument('--path', default='data/train', type=str,
#                     help='path of the root data foler')
# parser.add_argument('--smote', default=True, type=bool,
#                     help='whether using smote for data augmentation')
# parser.add_argument('--smote_k', default=5, type=int,
#                     help='The nearest neighbors used to define the neighborhood of samples to use to generate the synthetic samples')
# parser.add_argument('--size', default="(399,399)", type=tuple_int_type,
#                     help='size to resize')
# parser.add_argument('--bz', default=64, type=int,
#                     help='batch size')
# parser.add_argument('--norm_mean', default="(0.485,0.456,0.406)", type=tuple_float_type,
#                     help='Mean value of z-scoring normalization for each channel in image')
# parser.add_argument('--norm_std', default="(0.229,0.224,0.225)", type=tuple_float_type,
#                     help='Mean value of z-scoring standard deviation for each channel in image')
# parser.add_argument('--brightness', default="(0.8,2)", type=tuple_float_type,
#                     help='Brightness range for data augmentation')
# parser.add_argument('--noise_std', default=0.05, type=float,
#                     help='Adding noise with guassian distribution for data augmentation')
# parser.add_argument('--shuffle', default=True, type=bool,
#                     help='whether to shuffle training data during optimization')

# # Hardware Related
# parser.add_argument('--device_id', default=0, type=int,
#                     help='the id of the gpu to use')   

# # Training Arguments
# parser.add_argument('--model', default='resnet50', type=str,
#                     help='the model to use for training or make predictions')
# parser.add_argument('--reuse_model', default='', type=str,
#                     help='the name of the model to reuse for finetuning on new data')
# parser.add_argument('--optimizer', default='Adam', type=str,
#                     help='the optimizer to use')
# parser.add_argument('--lr', default=0.001, type=float,
#                     help='the optimizers learning rate')  
# parser.add_argument('--momentum', default=0.9, type=float,
#                     help='momentum factor for sgd')  
# parser.add_argument('--weight_decay', default=1e-4, type=float,
#                     help=' weight decay (L2 penalty) for both sgd and adam')  
# parser.add_argument('--freeze_num', default=7, type=int,
#                     help='number of layers to freeze during fine tuning')  
# parser.add_argument('--epochs', default=100, type=int,
#                     help='number of epochs')   
# parser.add_argument('--patience', default=100, type=int,
#                     help='patience for early stop')   

# # Ensembling Arguments
# parser.add_argument('--snapshot_ensemble', default=False, type=bool,
#                     help='whether snapshot ensembling to be performed')
# parser.add_argument('--estimators', default=10, type=int,
#                     help='number of estimators to be ensembled')  
# parser.add_argument('--voting', default='majority', type=str,
#                     help='how the ensembling between different estimators is performed')  

# # Deit related
# parser.add_argument('--teacher', default='deit_base_distilled_patch16_224', type=str,
#                     help='teacher model for deit')
# parser.add_argument('--student', default='deit_small_distilled_patch16_224', type=str,
#                     help='student model for deit')

# # Testing Model Related
# parser.add_argument('--test', default=False, type=lambda x: (str(x).lower() == 'true'),
#                     help='whether test a model and make prediction file')
# parser.add_argument('--test_path', default='data/test', type=str,
#                     help='path of the test data foler')
# parser.add_argument('--test_model', default='0.9787946428571429.pth', type=str,
#                     help='the model that will be used to produce predictions')