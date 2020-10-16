from predict import model_fn, predict_fn, input_fn, output_fn
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
    parser.add_argument('--img-height', type=int, default=128, metavar='N',
                        help='height of image (default: 128)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')

    args = parser.parse_args()
    model = model_fn(args.model_dir)
    while True:
        url = input("Please enter URL:\n")
        payload = {'url': url}
        payload_str = json.dumps(payload)
        print(payload_str)
        input_object = input_fn(payload_str)
        print(input_object)
        prediction = predict_fn(input_object, model)
        print(prediction)
        output = output_fn(prediction)
        print(output)