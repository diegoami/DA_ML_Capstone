from predict import model_fn
import argparse
import torch
import os
import json
import time
from train import get_data_loaders
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

def eval_model(model, criterion):
    use_gpu = torch.cuda.is_available()
    since = time.time()
    loss_test = 0
    acc_test = 0

    test_batches = len(dataloaders['val'])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders['val']):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        inputs, labels = data
        print(inputs.shape)
        print(inputs.numpy().shape)

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = torch.true_divide(loss_test, dataset_sizes['val'])
    avg_acc = torch.true_divide(acc_test, dataset_sizes['val'])

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

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


    parser.add_argument('--img-width', type=int, default=128, metavar='N',
                        help='width of image (default: 128)')
    parser.add_argument('--img-height', type=int, default=128, metavar='N',
                        help='height of image (default: 128)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    args = parser.parse_args()
    model = model_fn(args.model_dir)

    dataloaders, dataset_sizes, class_names = get_data_loaders(img_dir=args.data_dir, img_width=args.img_width, img_height=args.img_height, batch_size=args.batch_size )

    criterion = nn.CrossEntropyLoss()
    eval_model(model, criterion)