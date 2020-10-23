from predict import model_fn, predict_fn, input_fn, output_fn
import argparse
import os
import json
import numpy as np
from PIL import Image
import random
from sklearn.metrics import classification_report
from collections import defaultdict
from util import move_files_to_right_place

def verify(model, data_dir, percentage=1):
    """
    Give a classification report and a confidence matrix of a model
    :param model: the model to analyze
    :param data_dir: the directory containing data
    :param percentage: the percentage of data to analyze (0-1)
    :return:
    """
    label_index = 0

    acc_test = 0
    count = 0
    total = 0

    y_true, y_pred = [], []

    dirs = [s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))]
    move_files_to_right_place(class_names=class_names, data_dir=args.data_dir)
    np_conf = np.zeros((len(dirs), len(dirs)), dtype='uint')

    misclassified = defaultdict(list)
    for dir in dirs:

        curr_img_dir = os.path.join(data_dir, dir)
        images = os.listdir(curr_img_dir)
        for image in images:
            curr_img = os.path.join(curr_img_dir, image)
            total += 1
            rnd_value = random.uniform(0, 1)
            if (rnd_value > percentage):
                continue
            with open(curr_img, 'rb') as f:
                imagef = Image.open(f)
                prediction = predict_fn(imagef, model)
                output = output_fn(prediction)
                output_list = json.loads(output)
                pred_index, maxx = 0, -100
                for i, ol in enumerate(output_list):
                    if ol > maxx:
                        maxx = ol
                        pred_index = i

                acc_test += (pred_index == label_index)
                count += 1
                avg_acc = acc_test / count
                np_conf[label_index, pred_index] += 1
                y_true.append(label_index)
                y_pred.append(pred_index)
                if (pred_index != label_index and maxx > 1):
                    misckey = f'{label_index}:{pred_index}'
                    misclassified[misckey].append((image, maxx))
                    misclassified[misckey].sort(key=lambda x: x[1], reverse=True)

                if (count % 500 == 0):
                    print("{} processed up to {}".format(count, total))

        label_index += 1
    print("{} processed up to {}".format(count, total))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print("Confidence Matrix")
    print(np_conf)
    report = classification_report(y_true=y_true, y_pred=y_pred)
    print(report)

    return report, np_conf, misclassified

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
    model = model_fn(args.model_dir)
    report, np_conf, misclassified = verify(model, args.data_dir, 1)

    print(report)
    print(np_conf)
    print(misclassified)

    with open('misclassified.json', 'w', encoding='utf-8') as f:
        json.dump(misclassified, f, ensure_ascii=False, indent=4)

    for key in misclassified.keys():
        print(key)
        print("=============================")
        values = misclassified[key]
        for value in values:
            print(value)
