#!/bin/bash
python run.py --path data/train \
--smote true \
--smote_k 10 \
--size "(224,224)" \
--bz 32 \
--norm_mean "(0.485,0.456,0.406)" \
--norm_std "(0.229,0.224,0.225)" \
--brightness "(1,2)" \
--noise_std 0.5 \
--shuffle false \





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

