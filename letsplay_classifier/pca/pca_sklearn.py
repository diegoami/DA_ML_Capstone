import argparse
import os
import json
import numpy as np
from PIL import Image
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from pca.pca_commons import get_labels, do_pca, df_from_pca, plot_2d_pca, plot_3d_pca




def retrieve_df(data_dir, model_dir, width=80, height=45, mode='L', percentage=1, do_save=False):
    """
    retrieves a dataframe containing all images of the dataset in a 80x45, black and white format, in a numpy array, and their labels
    :param data_dir: images location
    :param model_dir: model directory where to save the dataframe
    :param width: width of image to resize to
    :param height: height of image to resize to
    :param mode: image mode to use to
    :param percentage: the percentage of image to build the dataframe with
    :return:
    """
    file_name_complete = f'fullnpy_{width}_{height}_{mode}_{percentage}.npy'
    full_name_complete = os.path.join(model_dir, file_name_complete)
    # goes through labels
    label_index = 0
    # images for which a prediction was made
    images_processed = 0
    images_total = 0
    X, y = None, None
    # directories and label names, sorted alphabetically
    dirs = get_labels(data_dir)
    # loop through all directories
    # ry / label names
    if do_save and os.path.isfile(full_name_complete):
        print(f'Reading from {full_name_complete }')
        with open(full_name_complete, 'rb') as f:
            df = np.load(f)
            X, y = df[:, :-1], df[:, -1]
    else:
        print(f'{full_name_complete} not found')
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
                        image = image.resize((width, height))
                        if (mode == 1):
                            image = image.convert(mode)
                        data = np.asarray(image).flatten()

                        if X is None:
                            X = data
                            y = np.array([label_index])
                        else:
                            X = np.vstack([X, data])
                            y = np.hstack([y, np.array(label_index)])
                        if (images_processed % 500 == 0):
                            print("{} processed up to {}".format(images_processed, images_total))

            label_index += 1

        df = np.hstack([X, np.expand_dims(y, axis=1)])
        if do_save:
            print(f'Saving to {full_name_complete}')
            os.makedirs(model_dir, exist_ok=True)
            with open(full_name_complete, 'wb') as f:
                np.save(f, df)
    return X, y

def do_estimator_wf(X, y, clf):
    print("=====================================")
    print(clf)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True, stratify=y)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    show_metrics(y_test, y_test_pred)
    print("=====================================")


def show_metrics(y_true, y_pred):
    accuracy_res = accuracy_score(y_true, y_pred)
    f1_score_res = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy_res}')
    print(f'F1 Score: {f1_score_res}')


    report = classification_report(y_true=y_true, y_pred=y_pred)
    print(report)
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--n-components', type=int, default=50,
                        help='number of PCA_Components')
    parser.add_argument('--img-width', type=int, default=80,
                        help='image width')
    parser.add_argument('--img-height', type=int, default=45,
                        help='image height')
    parser.add_argument('--img-mode', type=str, default='L',
                        help='image mode')
    parser.add_argument('--percentage', type=int, default=100,
                        help='percentage')


    args = parser.parse_args()

    print(f'Data Dir: {args.data_dir}')
    print(f'Model Dir: {args.model_dir}')

    X, y = retrieve_df(args.data_dir, args.model_dir, args.img_width, args.img_height, args.img_mode, args.percentage / 100, True)
    Xp = do_pca(X, args.n_components)
    df = df_from_pca(Xp, y, get_labels(args.data_dir))

    do_estimator_wf(Xp, y, RandomForestClassifier(n_estimators=100))
    do_estimator_wf(Xp, y, SGDClassifier(max_iter=1000, tol=1e-3))

    class_names = get_labels(args.data_dir)
    plot_2d_pca(df, class_names)
    plot_3d_pca(df)
