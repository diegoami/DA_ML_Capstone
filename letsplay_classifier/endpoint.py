
import argparse
import os
import json
import torch
from PIL import Image
import requests
import torch.nn as nn
import numpy as np
import random

from sagemaker.predictor import RealTimePredictor

def evaluate(endpoint_name, data_dir, percentage=1):
    """
    Does an evaluation on a subset of the images on the endpoint. Not great performance.
    
    endpoint_name : the name of the endpoint where 
    data_dir : the local directory where files can be found
    percentage : the percentage of files 
    """
    criterion = nn.CrossEntropyLoss()
    index = 0
    loss_test = 0
    acc_test = 0
    count = 0
    image_total = 0
    predictor = RealTimePredictor(endpoint_name,
         content_type='application/json',
         accept='application/json')

    dirs = sorted(os.listdir(data_dir))
    for dir in dirs:
        labels = torch.empty(1, dtype=int)
        labels[0] = index
        print(labels)
        curr_img_dir = os.path.join(data_dir, dir)
        images = os.listdir(curr_img_dir)
        for image in images:
            curr_img = os.path.join(curr_img_dir, image)
            image_total += 1
            if (random.uniform(0, 1) > percentage):
                continue
            with open(curr_img, 'rb') as f:
                imagef = Image.open(f)
                image_data = json.dumps(np.array(imagef).tolist())
                
                output = predictor.predict(image_data)
                
                output_list = json.loads(output)
                prediction = torch.FloatTensor(output_list).unsqueeze(0)
                print(prediction)
                _, preds = torch.max(prediction.data, 1)
                loss = criterion(prediction, labels)

                loss_test += loss.data
                acc_test += torch.sum(preds == labels.data)
                count += 1
                avg_loss = torch.true_divide(loss_test, count)
                avg_acc = torch.true_divide(acc_test, count)
                if (count % 50 == 0):
                    print("{} processed of {}".format(count, image_total))
                    print("Avg loss (test): {:.4f}".format(avg_loss))
                    print("Avg acc (test): {:.4f}".format(avg_acc))
        index += 1
    return avg_acc, avg_loss, count