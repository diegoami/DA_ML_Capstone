from __future__ import print_function # future proof
import argparse
import time
import copy
import os
import json
from sklearn.model_selection import train_test_split
from constants import IMG_WIDTH, IMG_HEIGHT

LOCAL = 1

import torch.utils.data

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
import torchdata as td
import torchvision
from torchvision import transforms

# import model
from model import VGGLP


def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGLP(model_info.num_classes)

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)



# Provided train function
def train(model, train_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero accumulated gradients
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

    # save trained model, after all epochs
    save_model(model, args.model_dir)


# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)
    
def save_model_params(num_classes):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'num_classes': num_classes,
            'img_width': args.img_width,
            'img_height': args.img_height
        }
        torch.save(model_info, f)

def get_data_loaders(img_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=8):

    root = img_dir
    total_count = sum([len(files) for r, d, files in os.walk(root)])

    data_transform = torchvision.transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]
    )
    model_dataset = td.datasets.WrapDataset(torchvision.datasets.ImageFolder(root, transform=data_transform))
    # Also you shouldn't use transforms here but below
    train_count = int(0.75 * total_count)
    valid_count = total_count - train_count
    import numpy as np

    train_idx, valid_idx = train_test_split(
        np.arange(len(model_dataset)),
        test_size=0.25,
        shuffle=True,
        stratify=model_dataset.targets)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    train_dataset_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_dataset_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size, sampler=valid_sampler)




    dataloaders = {
        'train': train_dataset_loader,
        'val': valid_dataset_loader
    }
    dataset_sizes = {
        'train': train_count,
        'val': valid_count
    }
    class_names = model_dataset.classes
    return dataloaders, dataset_sizes, class_names

def \
        train_model(vgg, dataloaders, dataset_sizes, criterion, optimizer,  num_epochs=15):
    use_gpu = torch.cuda.is_available()
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['val'])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(dataloaders['train']):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches ), end='', flush=True)



            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        print()

        avg_loss = torch.true_divide(loss_train, dataset_sizes['train'])
        avg_acc = torch.true_divide(acc_train, dataset_sizes['train'])

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(dataloaders['val']):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = torch.true_divide(loss_val, dataset_sizes['val'])
        avg_acc_val = torch.true_divide(acc_val, dataset_sizes['val'])

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Training Parameters, given

    parser.add_argument('--img-width', type=int, default=IMG_WIDTH, metavar='N',
                        help='width of image (default: 128)')
    parser.add_argument('--img-height', type=int, default=IMG_HEIGHT, metavar='N',
                        help='height of image (default: 72)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # get train loader
    #train_loader = _get_train_loader(args.batch_size, args.data_dir) # data_dir from above..

    dataloaders, dataset_sizes, class_names = get_data_loaders(img_dir=args.data_dir, img_width=args.img_width, img_height=args.img_height, batch_size=args.batch_size )

    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate

    model = VGGLP(len(class_names))
    if torch.cuda.is_available():
        model.cuda()

    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=args.epochs)
    save_model(model, args.model_dir)

    # Given: save the parameters used to construct the model
    save_model_params(num_classes=len(class_names))


