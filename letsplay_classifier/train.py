from __future__ import print_function # future proof
import argparse
import time
import copy
import os
import json
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import VGGLP

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


import torchdata as td
import torchvision
from torchvision import transforms


def save_model(model, model_dir):
    """
    saves the model.
    model - the model to be saved
    model_dir - the directory where to save it
    """
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)
    
def save_model_params(model_dir, class_names, img_width, img_height, epochs, layer_cfg):
    """
    Save the parameters using for creating the model
    :param model_dir : where to save the model
    :param classes : categories
    :param img_width: the width to which resize images
    :param img_height: the height to which resize images
    :param epochs: number of epochs in iteration
    :param layer_cfg: what configuration of layers to use in the VGG model ()
    """    
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'class_names': class_names,
            'img_width': img_width,
            'img_height': img_height,
            'epochs': epochs,
            'layer_cfg': layer_cfg
        }
        torch.save(model_info, f)

def save_model_metrics(model_dir, best_acc, best_loss):
    """
    Save the metrics found while training
    :param model_dir: where to save the model
    :param best_acc: best accuracy to save
    :param best_loss: best loss to save
    """
    model_metrics_path = os.path.join(model_dir, 'model_metrics.pth')
    with open(model_metrics_path, 'wb') as f:
        model_metrics = {
            'best_acc': best_acc,
            'best_loss': best_loss
        }
        torch.save(model_metrics, f)
        
        
def get_data_loaders(img_dir, img_height, img_width, batch_size=8):
    """
    Builds the data loader objects for retrieving images from a specific directory
    :param img_dir - the directory where images are located
    :param img_height - the height to which to compress images
    :param img_width - the width to which compress images
    :returns - the data loaders, the daset sizes, and the names of the labels
    """
    total_count = sum([len(files) for r, d, files in os.walk(img_dir)])

    data_transform = torchvision.transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    # build a dataset of images from the img_dir directory
    im_folder = torchvision.datasets.ImageFolder(img_dir, transform=data_transform)
    model_dataset = td.datasets.WrapDataset(im_folder)



    # stratify the dataset
    train_idx, valid_test_idx = train_test_split(
        np.arange(len(model_dataset)),
        test_size=0.35,
        shuffle=True,
        stratify=model_dataset.targets)



    valid_idx, test_idx = train_test_split(
        np.array(valid_test_idx),
        test_size=0.4,
        shuffle=True,
        stratify=np.array(model_dataset.targets)[valid_test_idx])

    train_count, valid_count, test_count = len(train_idx), len(valid_idx), len(test_idx)

    # create two data loaders for training and validation dataset
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_dataset_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_dataset_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size, sampler=valid_sampler)
    test_dataset_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size, sampler=test_sampler)
    all_dataset_loader = torch.utils.data.DataLoader(model_dataset)

    dataloaders = {
        'train': train_dataset_loader,
        'val': valid_dataset_loader,
        'test': test_dataset_loader,
        'all': all_dataset_loader
    }
    dataset_sizes = {
        'train': train_count,
        'val': valid_count,
        'test': test_count,
        'all': total_count
    }
    class_names = model_dataset.classes
    return dataloaders, dataset_sizes, class_names


def test_model(model,  criterion, test_data_loader, test_count):
    """
    executes a test of the model on the holdout set
    :param model: the model to test
    :param criterion: the criterion to use to optimize
    :param test_data_loader: the data loader for test data
    :param test_count: the amount of datapoints in the test dataset
    :return: predictions array, true values so that statistics can be generated
    """
    use_gpu = torch.cuda.is_available()
    since = time.time()
    loss_test = 0
    acc_test = 0
    y_pred, y_true = [], []
    test_batches = len(test_data_loader)
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(test_data_loader):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)
        y_pred = y_pred + list(preds.detach().cpu().numpy())
        y_true = y_true + list(labels.data.detach().cpu().numpy())

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = torch.true_divide(loss_test, test_count)
    avg_acc = torch.true_divide(acc_test, test_count)

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)
    return y_pred, y_true

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer,  num_epochs=10):
    """
    trains the model using a set of images
    :param model - the model to be trained
    :param dataloaders - a map of dataloaders for retrieving load images for train and val-idation
    :param dataset_sizes - a map of int containing the size of datasets for train and validation
    :param criterion - the criterion used to evaluate the model
    :param optimizer - the optimizer used to train the model
    :param num_epoch - the number of epochs to train
    :returns the trained model, accuracy and loss
    """
    use_gpu = torch.cuda.is_available()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1.0
    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['val'])

    # in each epoch
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        # first step - train the model on the training set
        model.train(True)
        for i, data in enumerate(dataloaders['train']):
            if i % 100 == 0:
                print("Training batch {}/{}".format(i, train_batches ))

            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
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

        # evaluate on the validation set
        model.train(False)
        model.eval()

        for i, data in enumerate(dataloaders['val']):
            if i % 100 == 0:
                print("Validation batch {}/{}".format(i, val_batches))

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            optimizer.zero_grad()

            outputs = model(inputs)
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
            best_loss = avg_loss_val
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, best_acc, best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker parameters

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--img-width', type=int, default=320, metavar='N',
                        help='width of image (default: 320)')
    parser.add_argument('--img-height', type=int, default=180, metavar='N',
                        help='height of image (default: 180)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--layer-cfg', type=str, default='D', metavar='N',
                        help='layer type for VGG')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    print(f'Data Dir: {args.data_dir}')
    print(f'Model Dir: {args.model_dir}')
    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    class_names = sorted(os.listdir(args.data_dir))


    # retrieves train and validation data loaders and datasets, and the label names         
    dataloaders, dataset_sizes, class_names = get_data_loaders(img_dir=args.data_dir,  img_height=args.img_height, img_width=args.img_width, batch_size=args.batch_size )

    # initializes a VGG model returning the desired labels
    model = VGGLP(len(class_names), args.layer_cfg)
    if torch.cuda.is_available():
        model.cuda()

    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # retrieves the trained model, along with best accuracy and loss
    model, best_acc, best_loss = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=args.epochs)



    save_model(model, args.model_dir)

    #  save the parameters used to construct the model
    save_model_params(args.model_dir, class_names, args.img_width, args.img_height, args.epochs, args.layer_cfg)

    # saves the best accuracy and loss in this model
    save_model_metrics(args.model_dir, best_acc, best_loss)

    y_pred, y_true = test_model(model, criterion, dataloaders['test'], dataset_sizes['test'])
    report = classification_report(y_true=y_true, y_pred=y_pred)
    print(report)


    print(confusion_matrix(y_true, y_pred))


