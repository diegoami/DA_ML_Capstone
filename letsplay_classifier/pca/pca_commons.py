import os
import pandas as pd

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def get_labels(data_dir):
    dirs = [s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))]
    return dirs

def do_pca(X, n_components=3):
    pca = PCA(n_components)
    pca_result = pca.fit_transform(X)
    return pca_result

def df_from_pca(X, y, labels):
    df = pd.DataFrame()
    palette = sns.color_palette("hls", len(labels))
    df['y'] = y
    df['label'] = df.apply(lambda x: labels[x['y']], axis=1)
    df['pca-one'] = X[:, 0]
    df['pca-two'] = X[:, 1]
    df['pca-three'] = X[:, 2]
    df["colors"] = df.apply(lambda x: palette[x['y']], axis=1)
    return df

def plot_3d_pca(df, labels):
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    palette = sns.color_palette("hls", len(labels))
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

def plot_2d_pca(df, classes):
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="label",
        palette=sns.color_palette("hls", len(classes)),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()