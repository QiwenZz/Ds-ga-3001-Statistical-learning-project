from torchvision import models
import torchvision
import torch
import timm


def load_model(model_name, args, device):
    if args['reuse_model'] != '':
        state = torch.load('models/'+args['reuse_model']) 
        model_name = state['args']['model']
        
    if model_name[:6] == 'resnet':
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        elif model_name == 'inception':
            model = models.inception_v3(pretrained=True)


        for c, child in enumerate(model.children()):
            if c<args['freeze_num']:
                for param in child.parameters():
                    param.requires_grad = False

        num_ftrs = model.fc.in_features

        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 12)
        )

        # loads in the weight for an existent model
        if args['reuse_model'] != '':
            model.load_state_dict(state['net'])
            print('loaded')
        '''
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 1200),
            torch.nn.BatchNorm1d(1200),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1200, 12)
        )
        '''
    elif model_name == 'deit':
        # Load the teacher model
        teacher_model = timm.create_model(args['teacher'], pretrained=True)
        # Freeze the teacher model's parameters
        for param in teacher_model.parameters():
            param.requires_grad = False

        # Load the student model
        student_model = timm.create_model(args['student'], pretrained=True)
        n_inputs = student_model.head.in_features
        student_model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 12)
        )

        teacher_model.to(device)
        student_model.to(device)
        return teacher_model, student_model
    model.to(device)
    return model

    