import argparse
from collections import defaultdict
import os
import json
import numpy as np
from PIL import Image
import random
from sklearn.metrics import classification_report

from util import  arg_max_list

THRESHOLD = 2.5
from predict import model_fn, predict_fn, output_fn

def verify(model, data_dir, percentage=1):
    """
    uses a model to predict the categories of a dataset, compare them with the true values and to return appropriate reports
    :param model: the model to analyze
    :param data_dir: the directory containing data
    :param percentage: the percentage of data to analyze (0-1)
    :return: a classification report, a confusion matrix, a map of possibly misclassified data points
    """

    # goes through labels
    label_index = 0

    # sum of accuracy of all predictions. Makes sense only when averaged at the end.
    acc_sum = 0

    # images for which a prediction was made
    images_processed = 0

    images_total = 0

    # directories and label names, sorted alphabetically
    dirs = [s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))]

    # confusion matrix in numpy format
    np_conf_matrix = np.zeros((len(dirs), len(dirs)), dtype='uint')

    # true values and predictions
    y_true, y_pred = [], []

    dubious_preds = []

    # loop all directory / label names
    for dir in dirs:
        curr_img_dir = os.path.join(data_dir, dir)
        images = os.listdir(curr_img_dir)

        # loop on all images in a directory, belonging to a label
        for image_index, image in enumerate(images):
            curr_img = os.path.join(curr_img_dir, image)
            images_total += 1

            # only for a given percentage of images
            if (random.uniform(0, 1) <= percentage):
                with open(curr_img, 'rb') as f:
                    images_processed += 1

                    # goes through predict_fn and output_fn in predict, but only using the model
                    image_data = Image.open(f)
                    prediction = predict_fn(image_data, model)
                    output_json = output_fn(prediction)

                    # prediction in log probabilities as output from the last step
                    pred_output = json.loads(output_json)

                    pred_index = arg_max_list(pred_output)

                    # the log probability of a prediction
                    label_log_prob = pred_output[label_index]

                    if (label_log_prob < THRESHOLD):
                        dubious_preds.append((curr_img, label_log_prob))


                    # comparing predictions and labels, updating metrics, confidence metrics and classification report
                    np_conf_matrix[label_index, pred_index] += 1
                    y_true.append(label_index)
                    y_pred.append(pred_index)


                    if (images_processed % 500 == 0):
                        print("{} processed up to {}".format(images_processed, images_total))

        label_index += 1

    report = classification_report(y_true=y_true, y_pred=y_pred)
    dubious_preds.sort(key=lambda x: x[1] )
    return report, np_conf_matrix, dubious_preds

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    print(f'Data Dir: {args.data_dir}')
    print(f'Model Dir: {args.model_dir}')

    model = model_fn(args.model_dir)
    report, np_conf_matrix, dubious_preds = verify(model, args.data_dir, 1)

    print("Confusion Matrix")
    print(np_conf_matrix)
    print(report)
    print(dubious_preds)


