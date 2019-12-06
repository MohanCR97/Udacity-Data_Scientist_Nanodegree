import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import os
import sys
import argparse


#building model function
def model_build(arch, hidden_units):
    # building network
    model = getattr(models, arch)
    model = model(pretrained=True)    
    for param in model.parameters():
        param.requires_grad = False
    dropout_rate = 0.5
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=dropout_rate)),
                                            ('output', nn.Linear(hidden_units, 102))]))
    model.classifier = classifier
    print('model building completed')
    return model

#training model fuction
def train_model(model, train_loader, valid_loader, learning_rate, gpu, epochs):
    if gpu and torch.cuda.is_available():
        device = 'cuda'
        print('use gpu to train')
    elif gpu == True and torch.cuda.is_available() == False:
        device = 'cpu'
        print('gpu is not available, use cpu to train')
    else:
        device = 'cpu'
        print('use cpu to train')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    
    # epoch start
    for e in range(epochs):
        training_loss = 0
        valid_loss = 0
        valid_accuracy = 0
        
        # training
        start = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            _,predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += float(loss)
            total = len(train_loader)
        print('Epoch: {}/{}'.format(e+1, epochs),
              'Training Loss: {:.3f}'.format(training_loss/total))
        print('Training time per epoch: {:.3f} seconds'.format(time.time()-start))
        
        #validation
        start = time.time()
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model.forward(images)
                valid_loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                equality = (labels.data == predicted)
                valid_accuracy += equality.type_as(torch.FloatTensor()).mean()
                total = len(valid_loader)
        print('Test Loss: {:.3f}'.format(valid_loss/total),
              'Test Accuracy: {:.3f}'.format(valid_accuracy/total))
        print('Testing time per epoch: {:.3f} seconds'.format(time.time()-start))
        model.train()
    print('model training completed')
    return optimizer

#testing model function
def test_model(model, test_loader, gpu):
    if gpu and torch.cuda.is_available():
        device = 'cuda'
        print('use gpu to test')
    elif gpu == True and torch.cuda.is_available() == False:
        device = 'cpu'
        print('gpu is not available, use cpu to test')
    else:
        device = 'cpu'
        print('use cpu to test')
    model.to(device)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    print('model testing completed')
    print('Accuracy on the test data: %d %%' % (100*correct/total))
    
#saing the model function
def save_model(save_dir, model, arch, train_datasets, optimizer):
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': train_datasets.class_to_idx,
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, '{}checkpoint.pth'.format(save_dir))
    print('model saving completed')
    
# Initializing directory function
def init_dir(dir):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except Exception as e:
            sys.stderr.write('[Error][%s]' % (e))
            sys.stderr.flush()
            return False
    return True

# main
if __name__ == '__main__':
    # read par
    parser = argparse.ArgumentParser(description='This program is used to train a model')
    parser.add_argument('data_dir', help = 'Training Set Data Dir', default = './flowers/')
    parser.add_argument('--save_dir', help = 'Set directory to save checkpoints', default = '')
    parser.add_argument('--arch', help = 'Arch can be choose from \"vgg19|vgg16|vgg16_bn\" or use default vgg16', default = 'vgg16')
    parser.add_argument('--learning_rate', help = 'Learning Rate', default = 0.001, type = float)
    parser.add_argument('--hidden_units', help = 'Hidden Units', default = 1024, type = int)
    parser.add_argument('--epochs', help = 'Epoches', default = 10, type = int)
    parser.add_argument('--gpu', help = 'Use GPU to Train', action = 'store_true', default = True)
    args = parser.parse_args()
    
    # print introduction
    print('***************************************')
    print('*This program is used to train a model*')
    print('***************************************')
    print('---------------parameters--------------')
    print('Training Set Data Dir:', args.data_dir)
    print('Checkpoints Saving Dir:', args.save_dir)
    print('Arch:', args.arch)
    print('Learning Rate:', args.learning_rate)
    print('Hidden Units:', args.hidden_units)
    print('Epoches:', args.epochs)
    print('Use GPU:', args.gpu)
    print('-------------Let\'s begin--------------')
    
    # check save dir
    if (args.save_dir != '') and init_dir(args.save_dir):
        print('Could not create dir')
        sys.exit(1)
        
    # check arch
    arch_list = ['vgg19', 'vgg16', 'vgg16_bn']
    if args.arch not in arch_list:
        print('Only support {}'.format(arch_list))
        sys.exit(2)
        
    # parameters set
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    batch_size = 64
    
   
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    print('data loader building completed')
    
    # build model
    model = model_build(args.arch, args.hidden_units)
    
    # train model
    optimizer = train_model(model, train_loader, valid_loader, args.learning_rate,
                            args.gpu, args.epochs)
    
    # test model on testset
    test_model(model, test_loader, args.gpu)
    
    # save model
    save_model(args.save_dir, model, args.arch, train_datasets, optimizer)