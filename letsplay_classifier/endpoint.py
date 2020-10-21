
import argparse
import os
import json
import torch
from PIL import Image
import requests
from math import exp, log
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
    index = 0
    loss_tot = 0
    acc_tot = 0
    count = 0
    image_total = 0
    predictor = RealTimePredictor(endpoint_name,
         content_type='application/json',
         accept='application/json')

    dirs = sorted(os.listdir(data_dir))
    for dir in dirs:
        curr_img_dir = os.path.join(data_dir, dir)
        images = os.listdir(curr_img_dir)
        for image in images:
            curr_img = os.path.join(curr_img_dir, image)
            image_total += 1
            if (random.uniform(0, 1) > percentage):
                continue
            count += 1
            with open(curr_img, 'rb') as f:
                imagef = Image.open(f)
                image_data = json.dumps(np.array(imagef).tolist())
                
                output = predictor.predict(image_data)
                
                ol = output_list = json.loads(output)

                avg_loss, avg_acc = loss_tot / count, acc_tot / count
                
                if True:
         #       if (count % 50 == 0):
                    print("{} processed of {}".format(count, image_total))
                    print("Avg loss (test): {:.4f}".format(avg_loss))
                    print("Avg acc (test): {:.4f}".format(avg_acc))
        index += 1
    return avg_acc, avg_loss, count



if __name__ == '__main__':
    
    endpoint_name='DA-ML-endpoint'
    avg_acc, avg_loss, count = evaluate(endpoint_name, '../wendy_cnn_frames_data', 0.05)
    print("{} processed of {}".format(count, image_total))
    print("Avg loss : {:.4f}".format(avg_loss))
    print("Avg acc : {:.4f}".format(avg_acc))