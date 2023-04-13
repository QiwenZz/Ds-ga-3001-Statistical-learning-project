import matplotlib.pyplot as plt
def train_model(dataloaders, args):
    train_loader, val_loader = dataloaders
    
    # test for training
    print('training')
    img, label = next(iter(train_loader))
    print('batch X shape:',img.shape)
    print('batch y shape:',label.shape)
    print('first sample y:'label[0])
    print('showing fisrt sample X in image:')
    plt.imshow(img[0].T)
    plt.show()

    # test for training
    print('testing')
    img, label = next(iter(val_loader))
    print('batch X shape:',img.shape)
    print('batch y shape:',label.shape)
    print('first sample y:'label[0])
    print('showing fisrt sample X in image:')
    plt.imshow(img[0].T)
    plt.show()
    
    
    
    
    