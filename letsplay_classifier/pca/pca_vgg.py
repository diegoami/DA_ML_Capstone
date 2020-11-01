from __future__ import print_function # future proof
import argparse
import os
import json
import torch.utils.data
import torch
import torch.nn as nn
import numpy as np
import torchdata as td
import torchvision
from torchvision import transforms
from predict import model_fn, get_model_info


from pca.pca_commons import do_pca, df_from_pca, plot_2d_pca, plot_3d_pca

        
def get_data_loaders(img_dir, img_height, img_width, batch_size=8):
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
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    # build a dataset of images from the img_dir directory
    im_folder = torchvision.datasets.ImageFolder(img_dir, transform=data_transform)
    model_dataset = td.datasets.WrapDataset(im_folder)

    dataset_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size)

    return dataset_loader, total_count


def get_feature_matrix_from_dataset(dataloader, maxx=None):

    X, y = None, None
    for i, data in enumerate(dataloader):
        inputs, labels = data

        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)

        np_step = outputs.detach().cpu().numpy()
        if X is None:
            X = np_step
            y = np.array(labels)
        else:
            X = np.vstack([X, np_step])
            y = np.hstack([y, np.array(labels)])
        if maxx is not None and i > maxx:
            break
    return X, y


if __name__ == '__main__':
    """
    Prints a TNSE visualization
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--img-width', type=int, default=320, metavar='N',
                        help='width of image (default: 128)')
    parser.add_argument('--img-height', type=int, default=180, metavar='N',
                        help='height of image (default: 72)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')

    args = parser.parse_args()

    print(f'Data Dir: {args.data_dir}')
    print(f'Model Dir: {args.model_dir}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = torch.cuda.is_available()
    model = model_fn(args.model_dir)
    model_info = get_model_info(args.model_dir)
    args = parser.parse_args()

    # remove last fully-connected layer to get feaut
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier

    model.train(False)
    model.eval()

    # retrieves data loaders and datasets, and the label names
    dataloader, dataset_size = get_data_loaders(img_dir=args.data_dir,  img_height=args.img_height, img_width=args.img_width, batch_size=args.batch_size)

    X, y = get_feature_matrix_from_dataset(dataloader)

    Xp = do_pca(X, 3)
    df = df_from_pca(Xp, y, model_info['class_names'])

    plot_2d_pca(df, model_info['class_names'])
    plot_3d_pca(df)
