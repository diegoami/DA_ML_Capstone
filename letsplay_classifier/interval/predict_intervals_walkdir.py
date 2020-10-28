from predict import model_fn, predict_fn, output_fn, get_model_info
import argparse
import os
import json
import numpy as np
from interval.predict_intervals_utils import convert_to_intervals, get_short_classes
from PIL import Image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # this is actually the directory of the frames to predicted, not to be used for training
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])


    args = parser.parse_args()
    model = model_fn(args.model_dir)
    model_info = get_model_info(args.model_dir)
    print(model_info)
    class_names = model_info.get('class_names', ['Battle', 'Hideout', 'Other', 'Siege', 'Tournament'])
    short_classes = get_short_classes(class_names)
    work_dir = os.path.join(args.data_dir, 'uncategorized')

    images = sorted([s for s in os.listdir(work_dir)])

    # list of visualizations for each frame in the teddst dataset
    frame_visualizations = []

    for image_index, image in enumerate(images):
        curr_img = os.path.join(work_dir, image)

        with open(curr_img, 'rb') as f:
            image_data = Image.open(f)
            prediction = predict_fn(image_data, model)
            output_json = output_fn(prediction)
            pred_output = json.loads(output_json)
            out_exps = np.exp(pred_output)
            out_normalized = (out_exps / sum(out_exps) * 20) .astype(int)

            # visualization showing a breakdown of probabilities, what is shown in an image
            frame_visualization = ''.join([min(int(out_normalized[x]), 20) * short_classes[x] for x in range(0, 5)]).rjust(20, '_')
            frame_visualizations.append(frame_visualization)

    # converts visualization to scenes intervals
    convert_to_intervals(frame_visualizations, short_classes, class_names, True)