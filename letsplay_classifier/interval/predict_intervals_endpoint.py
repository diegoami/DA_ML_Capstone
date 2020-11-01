
import os
import json
import numpy as np

from .predict_intervals_utils import convert_to_intervals, get_short_classes
from PIL import Image

def evaluate(predictor, data_dir, class_names, print_lines=False):
    """
    Method used to show scenes intervals on sagemaker
    :param predictor: name of predictor to call
    :param data_dir: location of the frames to categorize
    :param class_names: names of categories
    :return:
    """
    work_dir = os.path.join(data_dir, 'uncategorized')
    # loop on all images in a directory, belonging to a label
    images = sorted([s for s in os.listdir(work_dir)])


    # loop on all images in a directory, belonging to a label
    short_classes = get_short_classes(class_names)
    frame_visualizations = []

    for image_index, image in enumerate(images):
        curr_img = os.path.join(work_dir, image)

        with open(curr_img, 'rb') as image_file:
            image = Image.open(image_file)
            data = np.asarray(image)

            output = predictor.predict(data)
            output_sv = output[0]
            out_exps = np.exp(output_sv)
            out_normalized = (out_exps / sum(out_exps) * 20) .astype(int)
            frame_visualization = ''.join([min(int(out_normalized[x]), 20) * short_classes[x] for x in range(0, 5)]).rjust(20,'_')
            frame_visualizations.append(frame_visualization)

    convert_to_intervals(frame_visualizations, short_classes, class_names, print_lines)
    
