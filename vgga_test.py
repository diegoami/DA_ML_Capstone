from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchdata as td
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from letsplay_classifier.model import VGGLP

BATCH_SIZE = 8
NUM_WORKER = 2
TRAIN, VAL, TEST = 'train', 'val', 'test'

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
IMG_HEIGHT, IMG_WIDTH = 256, 256




def get_data_loaders():
    root = 'wendy_cnn_frames_data'
    total_count = sum([len(files) for r, d, files in os.walk(root)])

    data_transform = torchvision.transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor()
        ]
    )
    model_dataset = td.datasets.WrapDataset(torchvision.datasets.ImageFolder(root, transform=data_transform))
    # Also you shouldn't use transforms here but below
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        model_dataset, (train_count, valid_count, test_count)
    )



    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
    )
    valid_dataset_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
    )

    dataloaders = {
        TRAIN: train_dataset_loader,
        VAL: valid_dataset_loader,
        TEST: test_dataset_loader
    }
    dataset_sizes = {
        TRAIN: train_count,
        VAL: valid_count,
        TEST: test_count
    }
    class_names = model_dataset.classes
    return dataloaders, dataset_sizes, class_names



def eval_model(vgg, criterion):
    since = time.time()
    loss_test = 0
    acc_test = 0

    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = torch.true_divide(loss_test, dataset_sizes[TEST])
    avg_acc = torch.true_divide(acc_test, dataset_sizes[TEST])

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


def train_model(vgg, dataloaders, dataset_sizes, criterion, optimizer,  num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

            # Use half training dataset
            if i >= train_batches / 2:
                break

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

        avg_loss = torch.true_divide(loss_train, dataset_sizes[TRAIN])
        avg_acc = torch.true_divide(acc_train, dataset_sizes[TRAIN])

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(dataloaders[VAL]):
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

        avg_loss_val = torch.true_divide(loss_val, dataset_sizes[VAL])
        avg_acc_val = torch.true_divide(acc_val, dataset_sizes[VAL])

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

dataloaders, dataset_sizes, class_names = get_data_loaders()
vgg16 = VGGLP(len(class_names))
if use_gpu:
    vgg16.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

vgg16 = train_model(vgg16, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=15)
torch.save(vgg16.state_dict(), 'VGG16_mblade.pt')
vgg16.load_state_dict(torch.load('VGG16_mblade.pt'))
eval_model(vgg16, dataloaders, dataset_sizes, criterion)