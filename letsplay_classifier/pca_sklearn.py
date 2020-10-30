import argparse
import os
import json
import numpy as np
from PIL import Image
import random
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


def do_pca(X, y, n_components=3):

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    Xpca = pca.transform(X)
    print(Xpca)
    return X, y


def retrieve_df(data_dir, model_dir, percentage):
    """
    retrieves a dataframe containing all images of the dataset in a 80x45, black and white format, in a numpy array, and their labels
    :param data_dir: images location
    :param model_dir: model directory where to save the dataframe

    :param percentage:
    :return:
    """
    file_name_complete = f'fullnpy_{percentage}.py'
    # goes through labels
    label_index = 0
    # images for which a prediction was made
    images_processed = 0
    images_total = 0
    X, y = None, None
    # directories and label names, sorted alphabetically
    dirs = [s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))]
    # loop all directo
    # ry / label names
    if os.path.isfile(os.path.join(model_dir, file_name_complete)):
        with open(os.path.join(model_dir, file_name_complete), 'rb') as f:
            df = np.load(f)
            X, y = df[:, :-1], df[:, -1]
    else:
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

                        image = Image.open(f)
                        image = image.resize((80, 45))
                        image = image.convert('1')
                        data = np.asarray(image).flatten()

                        if X is None:
                            X = data
                            y = np.array([label_index])
                        else:
                            X = np.vstack([X, data])
                            y = np.vstack([y, np.array(label_index)])
                        if (images_processed % 500 == 0):
                            print("{} processed up to {}".format(images_processed, images_total))

            label_index += 1
        df = np.hstack([X, y.unravel()])
        with open(os.path.join(model_dir, file_name_complete), 'wb') as f:
            np.save(f, df)
    return X, y

def do_estimator_wf(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    show_metrics(y_test, y_test_pred)


def show_metrics(y_true, y_pred):
    print(accuracy_score(y_true, y_pred))
    print(f1_score(y_true, y_pred, average='weighted'))

if __name__ == '__main__':

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

    X, y = retrieve_df(args.data_dir, args.model_dir, 1)
    Xp, yp = do_pca(X, y, 100)
    do_estimator_wf(Xp, yp, RandomForestClassifier(n_estimators=80))
    do_estimator_wf(Xp, yp, SGDClassifier(max_iter=1000, tol=1e-3))