import os
import torch
import torch.utils.data
import json

import requests
from PIL import Image
from torch.autograd import Variable
from model import VGGLP
from constants import IMG_HEIGHT, IMG_WIDTH
import numpy as np


from torchvision import datasets, models, transforms

def model_fn(model_dir_arg):
    global IMG_HEIGHT, IMG_WIDTH 
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")
    model_dir = '/opt/ml/model'
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)
    IMG_HEIGHT, IMG_WIDTH = model_info['img_height'], model_info['img_width']
    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGLP(model_info['num_classes'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


def input_fn(request_body, content_type='application/json'):
    """
    predictor using an image in its request
    """

    if content_type == 'application/json':
        image_data = Image.fromarray(np.array(json.loads(request_body), dtype='uint8'))
        image_resized = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))(image_data)
        image_tensor = transforms.ToTensor()(image_resized)
        image_unsqueezed = image_tensor.unsqueeze(0)
        return image_unsqueezed
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')


def output_fn(prediction_output, accept='application/json'):
    """
    result of a request as an array of probabilities in json format
    """
    
    print('Serializing the generated output.')
    if accept == 'application/json':
        arr = prediction_output.numpy()
        listresult = arr.flatten().tolist()
        print(listresult)
        json_res = json.dumps(listresult)
        print(json_res)
        return json_res
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    """
    executes a prediction based on a model
    input_data - the data point to predict (an image) as a pytorch 
    
    """
    print('Predicting class labels for the input data...')
    print(input_data.shape)
    if torch.cuda.is_available() :
        inputs = Variable(input_data.cuda(), volatile=True)
    else:
        inputs = Variable(input_data, volatile=True)
    print(inputs.shape)

    # Compute the result of applying the model to the input data.
    out = model(inputs)
    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach()

    return result