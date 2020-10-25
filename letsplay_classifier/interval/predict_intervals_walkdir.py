from predict import model_fn, predict_fn, output_fn, get_model_info
import argparse
import os
import json
import numpy as np
from constants import IMG_WIDTH, IMG_HEIGHT
from interval.predict_intervals_utils import convert_to_intervals, get_short_classes
from PIL import Image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--img-width', type=int, default=IMG_WIDTH, metavar='N',
                        help='width of image (default: 128)')
    parser.add_argument('--img-height', type=int, default=IMG_HEIGHT, metavar='N',
                        help='height of image (default: 72)')
    parser.add_argument('--layer-cfg', type=str, default='D', metavar='N',
                        help='layer type for VGG')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')


    args = parser.parse_args()
    model = model_fn(args.model_dir)
    model_info = get_model_info(args.model_dir)
    classes = model_info['class_names']
    short_classes = get_short_classes(classes)
    work_dir = os.path.join(args.data_dir, 'uncategorized')

    images = sorted([s for s in os.listdir(work_dir)])
    ev_seqs = []

    for image_index, image in enumerate(images):
        curr_img = os.path.join(work_dir, image)

        with open(curr_img, 'rb') as f:
            image_data = Image.open(f)
            prediction = predict_fn(image_data, model)
            output_json = output_fn(prediction)
            pred_output = json.loads(output_json)
            out_exps = np.exp(pred_output)
            out_work = (out_exps / sum(out_exps) * 20) .astype(int)
            categor_str = ''.join([min(int(out_work[x]), 20) * short_classes[x] for x in range(0, 5)]).rjust(20,'_')
            ev_seqs.append(categor_str)

    convert_to_intervals(ev_seqs, short_classes, classes)