from __future__ import print_function # future proof
import argparse

import os
import json

from constants import IMG_WIDTH, IMG_HEIGHT
import torch.utils.data
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchdata as td
import torchvision
from torchvision import transforms
from predict import model_fn, predict_fn, output_fn

        
        
def get_data_loaders(img_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=8):
    """
    Builds the data loader objects for retrieving images from a specific directory
    :param img_dir - the directory where images are located
    :param img_height - the height to which to compress images
    :param img_width - the width to which compress images
    :returns - the data loaders, the daset sizes, and the names of the labels
    """
    total_count = sum([len(files) for r, d, files in os.walk(img_dir)])

    data_transform = torchvision.transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]
    )
    
    # build a dataset of images from the img_dir directory
    im_folder = torchvision.datasets.ImageFolder(img_dir, transform=data_transform)
    model_dataset = td.datasets.WrapDataset(im_folder)

    dataset_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size)

    return dataset_loader, total_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker parameters


    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--img-width', type=int, default=IMG_WIDTH, metavar='N',
                        help='width of image (default: 128)')
    parser.add_argument('--img-height', type=int, default=IMG_HEIGHT, metavar='N',
                        help='height of image (default: 72)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')


    args = parser.parse_args()

    print(f'Data Dir: {args.data_dir}')
    print(f'Model Dir: {args.model_dir}')

    print(f'Data Dir: {args.data_dir}')
    print(f'Model Dir: {args.model_dir}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = torch.cuda.is_available()
    model = model_fn(args.model_dir)
    args = parser.parse_args()



    # remove last fully-connected layer
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier

    model.train(False)
    model.eval()


    # set the seed for generating random numbers




    # retrieves train and validation data loaders and datasets, and the label names         
    dataloader, dataset_size = get_data_loaders(img_dir=args.data_dir,  img_height=args.img_height, img_width=args.img_width, batch_size=args.batch_size)

    X = None
    y = None

    for i, data in enumerate(dataloader):
        #print(data)
        inputs, labels = data

        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)

        np_step = outputs.detach().cpu().numpy()
        if X is None:
            X = np_step
            y = labels
        else:
            X = np.vstack([X, np_step])
            y = np.hstack([y, labels])
        #print(np_total.shape)

    df = pd.DataFrame(X)

    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print(df[['pca-one', 'pca-two', 'pca-three', 'y']])
    #
    import matplotlib.pyplot as plt
    import seaborn as sns

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"],
        ys=df["pca-two"],
        zs=df["pca-three"],
        c=df["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()

    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="pca-one", y="pca-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 5),
    #     data=df,
    #     legend="full",
    #     alpha=0.3
    # )
    # plt.show()
    #