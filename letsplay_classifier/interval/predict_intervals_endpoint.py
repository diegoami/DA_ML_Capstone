
import os
import json
import numpy as np

from .predict_intervals_utils import convert_to_intervals, get_short_classes
from PIL import Image
from sagemaker.predictor import RealTimePredictor

def evaluate(endpoint_name, data_dir, class_names):
    """
    Method used to show scenes intervals on sagemaker
    :param endpoint_name: name of endpoint to call
    :param data_dir: location of the frames to categorize
    :param class_names: names of categories
    :return:
    """
    work_dir = os.path.join(data_dir, 'uncategorized')
    # loop on all images in a directory, belonging to a label
    images = sorted([s for s in os.listdir(work_dir)])


    # loop on all images in a directory, belonging to a label
    short_classes = get_short_classes(class_names)
    ev_seqs = []
    predictor = RealTimePredictor(endpoint_name,
                                  content_type='application/json',
                                  accept='application/json')

    for image_index, image in enumerate(images):
        curr_img = os.path.join(work_dir, image)

        with open(curr_img, 'rb') as f:
            image_data = Image.open(f)
            image_data = json.dumps(np.array(image_data).tolist())
            output_json = predictor.predict(image_data)
            pred_output = json.loads(output_json)
            out_exps = np.exp(pred_output)
            out_work = (out_exps / sum(out_exps) * 20) .astype(int)
            categor_str = ''.join([min(int(out_work[x]), 20) * short_classes[x] for x in range(0, 5)]).rjust(20,'_')
            ev_seqs.append(categor_str)

    convert_to_intervals(ev_seqs, short_classes, class_names)