import os
import pandas as pd

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def get_labels(data_dir):
    """
    retrieve labels in dataset
    :param data_dir: directory where the dataset is located
    :return: all labels in the dataset
    """
    dirs = [s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))]
    return dirs

def do_pca(X, n_components=3):
    """
    returns a PCA representation of a numpy array
    :param X: the array to apply PCA to
    :param n_components: how many PCA components
    :return: the PCA into which it has been decomposed
    """
    pca = PCA(n_components)
    pca_result = pca.fit_transform(X)
    return pca_result


def df_from_pca(X, y, labels):
    """
    creates a dataframe for visualization from numpy arrays
    :param X: features array
    :param y: labels array (as int)
    :param labels: string representation of labels
    a dataframe for visualization
    """
    df = pd.DataFrame()
    palette = sns.color_palette("hls", len(labels))
    df['y'] = y
    df['label'] = df.apply(lambda x: labels[x['y']], axis=1)
    df['pca-one'] = X[:, 0]
    df['pca-two'] = X[:, 1]
    df['pca-three'] = X[:, 2]
    df["colors"] = df.apply(lambda x: palette[x['y']], axis=1)
    return df

def plot_3d_pca(df):
    """
    plot pca on a 3d
    :param df: dataframe containing principal components
    """
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"],
        ys=df["pca-two"],
        zs=df["pca-three"],
        c=df["colors"],
        label=df["colors"]
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    ax.grid(True)
    plt.show()
    plt.savefig('3d_tse.png')

def plot_2d_pca(df, labels):
    """
    plot pca on a 3d
    :param df: dataframe containing principal components
    :param labels: name of the lables to show
    """
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="label",
        palette=sns.color_palette("hls", len(labels)),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()