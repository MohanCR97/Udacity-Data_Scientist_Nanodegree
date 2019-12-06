import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import models
import json
from PIL import Image
import os
import sys
import argparse


# loads a checkpoint and rebuilds the model func
def load_checkpoint(filepath, arch):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == arch:
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    else:
        print('Sorry.The arch dosen\'t match the checkpoint. You need to train a new model.')
        sys.exit(5)
    
    


# Image Preprocessing func
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # load image
    img = Image.open(image)
    # resize image
    length, width = img.size
    base = 1
    if length > width:
        base = 256 / width
    else:
        base = 256 / length
    img = img.resize((int(length * base), int(width * base)))
    # crop image
    length, width = img.size
    img = img.crop((length / 2 - 112, width / 2 - 112,
                    length / 2 + 112, width / 2 + 112))
    # image to np.array
    np_img = np.array(img) / np.array([225, 225, 225])
    # normalized
    normalized = (np_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    normalized = normalized.transpose()
    return normalized


# Image show func
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    #add a title
    ax.set_title(title)
    ax.imshow(image)
    
    return ax

# Class Prediction func
def predict(image_path, model, topk=5, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu and torch.cuda.is_available():
        device = 'cuda'
        print('use gpu to predict')
    elif gpu == True and torch.cuda.is_available() == False:
        device = 'cpu'
        print('gpu is not available, use cpu to predictt')
    else:
        device = 'cpu'
        print('use cpu to predict')
    model.to(device)
    image = torch.from_numpy(process_image(image_path))
    image = image.to(device)
    image = torch.tensor(image)
    image = image.float()
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
    output = output.to('cpu')
    pred = F.softmax(output, dim=1)
    probs, classes = torch.topk(pred, topk)
    probs = probs.data.numpy()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = classes.data.numpy()[0]
    top_classes = [idx_to_class[i] for i in classes]
    return probs, top_classes  


# check image file path
def check_img(dir):
    if not os.path.exists(dir):
        print('Image not exists')
        return False
    return True

# main
if __name__ == '__main__':
    # read par
    parser = argparse.ArgumentParser(description='This program uses the model to predict categories')
    parser.add_argument('image', help = 'Input Image File', default = False)
    parser.add_argument('checkpoint', help = 'Input Checkpoint File to Build A MODEL', default = False)
    parser.add_argument('--topk', help = 'Top k', default = 5, type = int)
    parser.add_argument('--category_names', help = 'Category Names File', default = 'cat_to_name.json')
    parser.add_argument('--gpu', help = 'Use GPU to Predict', action = 'store_true', default = True)
    parser.add_argument('--arch', help = 'Arch can be choose from \"vgg19|vgg16|vgg16_bn\" or use default vgg16', default = 'vgg16')
    args = parser.parse_args()

    # print introduction
    print('********************************************')
    print('*This program is used to predict categories*')
    print('********************************************')
    print('-----------------parameters-----------------')
    print('Input Image File:', args.image)
    print('Checkpoints File:', args.checkpoint)
    print('Arch:', args.arch)
    print('Category Names File:', args.category_names)
    print('Top k:', args.topk)
    print('Use GPU:', args.gpu)
    print('-----------------Let\'s begin---------------')
    
    # check image file path
    if check_img(args.image):
        print('Image loading completed')
    else:
        sys.exit(1)

        
    # check arch
    arch_list = ['vgg19', 'vgg16', 'vgg16_bn']
    if args.arch not in arch_list:
        print('Only support {}'.format(arch_list))
        sys.exit(2)
        
    # check and load checkpoint file
    if os.path.exists(args.checkpoint):
        try:
            model, optimizer = load_checkpoint(args.checkpoint, args.arch)
        except Exception as e:
            sys.stderr.write('[Error][%s]' % (e))
            sys.stderr.flush()
            sys.exit(3)
    else:
        print('Checkpoint file not exists')
        sys.exit(4)
        
    # load json
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # predict
    probs, classes = predict(args.image, model, args.topk, args.gpu)
    classes = [cat_to_name[c] for c in classes]
    
    # print result
    for i in zip(classes, probs):
        print(i)
        
    print('Prediction completed')