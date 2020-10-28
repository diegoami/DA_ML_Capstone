import os
import torch
import torch.utils.data
import json
from PIL import Image
from model import VGGLP
import numpy as np
import io

from torchvision import transforms

def get_model_info(model_dir_arg):
    """
    Gets model information (metadata)
    :param model_dir_arg: location of the model
    :return: full model information
    """
    model_dir = real_model_dir(model_dir_arg)

    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)
    return model_info

def real_model_dir(model_dir_arg):
    if (os.path.exists('/opt/ml/model')):
        model_dir = '/opt/ml/model'
    else:
        model_dir = model_dir_arg
    return model_dir

def model_fn(model_dir_arg):
    """
    Load the PyTorch model from the `model_dir` directory or from /opt/ml/model, in Sagemaker.
    :param model_dir_arg where the model is located
    """
    model_dir = real_model_dir(model_dir_arg)
    global IMG_HEIGHT, IMG_WIDTH


    model_info = get_model_info(model_dir)
    IMG_HEIGHT, IMG_WIDTH = model_info['img_height'], model_info['img_width']
    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = model_info.get('num_classes', len(model_info.get('class_names', [])))
    model = VGGLP(num_classes, model_info['layer_cfg'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


def input_fn(request_body, content_type='application/x-image'):
    """
    predictor accepting an image in json format and converting it into a Pillow Image
    :param request_body a request containing a PIL image in JSON
    :returns the image as a PIL image
    """

    if content_type == 'application/x-image':
        # converts images from json format
        image_data = Image.open(io.BytesIO(request_body))
        return image_data
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')


def output_fn(prediction_output, accept='application/json'):
    """
    result of a request as an array of probabilities in json format
    :param prediction_output : the prediction returned from the model
    """

    if accept == 'application/json':
        arr = prediction_output.numpy()
        listresult = arr.flatten().tolist()
        json_res = json.dumps(listresult)
        return json_res
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    """
    executes a prediction based on a model
    :param: input_data - a PIL image
    """
    image_resized = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))(input_data)
    image_tensor = transforms.ToTensor()(image_resized)
    image_unsqueezed = image_tensor.unsqueeze(0)
    inputs = image_unsqueezed.cuda() if torch.cuda.is_available() else image_unsqueezed
    # Compute the result of applying the model to the input data.
    out = model(inputs)
    result = out.cpu().detach()
    return result