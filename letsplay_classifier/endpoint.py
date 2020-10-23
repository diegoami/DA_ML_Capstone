import os
import json

from PIL import Image
import numpy as np
import random
from util import move_files_to_right_place

from sagemaker.predictor import RealTimePredictor

def evaluate(endpoint_name, data_dir, percentage=1):
    """
    Does an evaluation on a subset of the images on the endpoint.
    
    :param endpoint_name : the name of the endpoint where
    :param data_dir : the local directory where files can be found
    :param percentage : the percentage of image to check
    :return: list of predictions and true values
    """

    count = 0
    image_total = 0
    
    # label of images
    label_index = 0
    y_true, y_pred = [], []
    
    # set up a predictor from the endpoint
    predictor = RealTimePredictor(endpoint_name,
         content_type='application/json',
         accept='application/json')
    
    # we scan dirs in alphabetical orders, as the data loaders do
    dirs = [s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))]
    move_files_to_right_place(class_names=dirs, data_dir=data_dir)

    for dir in dirs:
        curr_img_dir = os.path.join(data_dir, dir)
        images = os.listdir(curr_img_dir)
        
        # scanning all images belonging to a label
        for image in images:
            curr_img = os.path.join(curr_img_dir, image)
            image_total += 1
            
            # we use only a random subset (performance)
            if (random.uniform(0, 1) > percentage):
                continue
            count += 1
            
            with open(curr_img, 'rb') as f:
                
                # pass image as a json
                imagef = Image.open(f)
                image_data = json.dumps(np.array(imagef).tolist())
                output = predictor.predict(image_data)
                output_list = json.loads(output)

                pred_index, maxx = 0, -100
                
                # "argmax" on a list to find the most likely label
                for i, ol in enumerate(output_list):
                    if ol > maxx:
                        maxx = ol
                        pred_index = i

                count += 1
                y_true.append(label_index)
                y_pred.append(pred_index)
                if (count % 50 == 0):
                    print("{} processed up to {}".format(count, image_total))
        label_index += 1
    return y_true,  y_pred



if __name__ == '__main__':
    
    endpoint_name='DA-ML-endpoint'
    avg_acc, avg_loss, count = evaluate(endpoint_name, '../wendy_cnn_frames_data', 0.05)
    print("{} processed of {}".format(count, image_total))
    print("Avg loss : {:.4f}".format(avg_loss))
    print("Avg acc : {:.4f}".format(avg_acc))