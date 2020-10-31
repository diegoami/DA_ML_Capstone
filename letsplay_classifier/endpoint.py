import io
import os
import json

from PIL import Image
import numpy as np
import random
from .util import arg_max_list

def evaluate(predictor, data_dir, percentage=1):
    """
    Does an evaluation on a subset of the images on the endpoint.
    
    :param endpoint_name : the name of the endpoint where
    :param data_dir : the local directory where files can be found
    :param percentage : the percentage of image to check
    :return: list of predictions and true values
    """

    images_processed = 0
    images_total = 0
    
    # label of images
    label_index = 0

    # true values and predictions
    y_true, y_pred = [], []

    
    # we scan dirs in alphabetical orders, as the data loaders do
    dirs = [s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))]

    for dir in dirs:
        curr_img_dir = os.path.join(data_dir, dir)
        images = os.listdir(curr_img_dir)
        
        # scanning all images belonging to a label
        for image in images:
            curr_img = os.path.join(curr_img_dir, image)
            images_total += 1
            
            # we use only a random subset (performance)
            if (random.uniform(0, 1) > percentage):
                continue
            images_processed += 1
            
            with open(curr_img, 'rb') as image_file:
                # retrive most likely category from predictor

                image = Image.open(image_file)
                data = np.asarray(image)

                output = predictor.predict(data)
                output_sv = output[0]
                pred_index = np.argmax(output_sv)

                images_processed += 1
                y_true.append(label_index)
                y_pred.append(pred_index)

                if (images_processed % 5000 == 0):
                    print("{} processed up to {}".format(images_processed, images_total))
        label_index += 1
    return y_true,  y_pred


