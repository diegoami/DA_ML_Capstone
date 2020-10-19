from predict import model_fn, predict_fn
import argparse
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn



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
    parser.add_argument('--img-height', type=int, default=72, metavar='N',
                        help='height of image (default: 72)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    args = parser.parse_args()
    model = model_fn(args.model_dir)
    criterion = nn.CrossEntropyLoss()
    index = 0
    loss_test = 0
    acc_test = 0
    count = 0

    dirs = sorted(os.listdir(args.data_dir))
    for dir in dirs:
        labels = torch.empty(1, dtype=int)
        labels[0] = index
        print(labels)
        curr_img_dir = os.path.join(args.data_dir, dir)
        images = os.listdir(curr_img_dir)
        for image in images:
            curr_img = os.path.join(curr_img_dir, image)
            print(curr_img)
            with open(curr_img, 'rb') as f:
                image_data = Image.open(f).resize((args.img_height, args.img_width))

                image_resized = transforms.Resize((args.img_height, args.img_width))(image_data)
                image_tensor = transforms.ToTensor()(image_resized)
                image_unsqueezed = image_tensor.unsqueeze(0)
                prediction = predict_fn(image_unsqueezed, model)

                _, preds = torch.max(prediction.data, 1)
                print(prediction)
                print(prediction.data)
                print(prediction.data.shape)

                print(preds)
                print(preds.shape)

                loss = criterion(prediction, labels)

                loss_test += loss.data
                acc_test += torch.sum(preds == labels.data)
                count += 1
                avg_loss = torch.true_divide(loss_test, count)
                avg_acc = torch.true_divide(acc_test, count)
                print("{} processed".format(count))
                print("Avg loss (test): {:.4f}".format(avg_loss))
                print("Avg acc (test): {:.4f}".format(avg_acc))
        index += 1