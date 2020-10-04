import os
import numpy as np
import torch
import torch.utils.data
import json
from six import BytesIO
import torchdata as td
import torchvision
from torchvision import datasets, models, transforms
import requests
from PIL import Image

# import model
from .model import VGGLP
IMG_HEIGHT, IMG_WIDTH = 128, 128
# accepts and returns numpy data
CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

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


    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']

        image_data = Image.open(requests.get(url, stream=True).raw)

        image_transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor()
        ])
        return image_transform(image_data)
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data.
    out = model(data)
    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach().numpy()

    return result