from predict import model_fn, predict_fn, input_fn, output_fn
import argparse
import os
import json
import numpy as np
from PIL import Image
import random
from sklearn.metrics import classification_report


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

    inc_index = 0
    loss_test = 0
    acc_test = 0
    count = 0
    total = 0

    y_true, y_pred = [], []


    dirs = sorted(os.listdir(args.data_dir))
    np_conf = np.zeros((len(dirs), len(dirs)))

    percentage = 0.05
    for dir in dirs:
        label_index =  inc_index

        curr_img_dir = os.path.join(args.data_dir, dir)
        images = os.listdir(curr_img_dir)
        for image in images:
            curr_img = os.path.join(curr_img_dir, image)
            total += 1
            if (random.uniform(0, 1) > percentage):
                continue
            with open(curr_img, 'rb') as f:

                imagef = Image.open(f)
                image_data = json.dumps(np.array(imagef).tolist())
                input_object = input_fn(image_data )

                prediction = predict_fn(input_object, model)

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
                np_conf[ label_index, pred_index] += 1
                y_true.append(label_index)
                y_pred.append(pred_index)
                if (count % 50 == 0):
                    print("{} processed up to {}".format(count, total))
                    print("Avg acc (test): {:.4f}".format(avg_acc))
                    print(np_conf)
        inc_index += 1
    print("{} processed up to {}".format(count, total))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print(np_conf)
    report = classification_report(y_true=y_true, y_pred=y_pred)
    print(report)