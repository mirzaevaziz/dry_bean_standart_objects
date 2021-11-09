from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
markers = {"Noisy object": "o", "Simple object": "X", 'Standart object': '^'}


def scale(dataFrame):
    result = dataFrame.copy()
    for feature_name in dataFrame.columns:
        if feature_name != 'Class':
            max_value = dataFrame[feature_name].max()
            min_value = dataFrame[feature_name].min()
            result[feature_name] = (
                dataFrame[feature_name] - min_value) / (max_value - min_value)
        else:
            result[feature_name] = dataFrame[feature_name]
    return result


def show_pca(dataFrame):
    dataFrame = dataFrame.copy()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(dataFrame[dataFrame.columns[:-2]].values)
    print(pca_result)

    dataFrame['pca-one'] = pca_result[:, 0]
    dataFrame['pca-two'] = pca_result[:, 1]

    plt.figure(figsize=(3, 2))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="Class",
        style="Type",
        palette=sns.color_palette("hls", len(np.unique(df['Class']))),
        data=dataFrame,
        legend="full",
        alpha=1,
        markers=markers
    )

    plt.show()


def show_tsne(dataFrame):
    dataFrame = dataFrame.copy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
    tsne_results = tsne.fit_transform(dataFrame[dataFrame.columns[:-2]].values)

    dataFrame['tsne-one'] = tsne_results[:, 0]
    dataFrame['tsne-two'] = tsne_results[:, 1]
    plt.figure(figsize=(3, 2))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="Class",
        style="Type",
        palette=sns.color_palette("hls", len(np.unique(df['Class']))),
        data=dataFrame,
        legend="full",
        alpha=1,
        markers=markers
    )

    plt.show()


while(True):
    df = pd.read_csv("Dry_Bean.txt", sep='\t')
    noisyObjects = np.loadtxt('data_noisy_objetcs.txt').astype(int)
    lcStandartObjects = np.loadtxt(
        'data_local_metric_standart_objects.txt').astype(int)
    nlcStandartObjects = np.loadtxt(
        'data_non_local_metric_standart_objects.txt').astype(int)

    # df['Class'][noisyObjects] = df['Class'][noisyObjects] + '_N'
    df = scale(df)
    df.loc[:, 'Type'] = 'Simple object'
    df.loc[noisyObjects, 'Type'] = 'Noisy object'

    print("Do you want to show local metric? (y/[n])", end='')
    answer = input()
    if answer.lower() == 'y':
        df.loc[lcStandartObjects, 'Type'] = 'Standart object'
        print("Do you want to remove simple objects? (y/[n])", end='')
        answer = input()
        if answer.lower() == 'y':
            df = df.loc[lcStandartObjects, :]
        else:
            print("Do you want to remove noisy objects? (y/[n])", end='')
            answer = input()
            if answer.lower() == 'y':
                df = df.drop(noisyObjects)
    else:
        df.loc[nlcStandartObjects, 'Type'] = 'Standart object'
        print("Do you want to remove simple objects? (y/[n])", end='')
        answer = input()
        if answer.lower() == 'y':
            df = df.loc[nlcStandartObjects, :]
        else:
            print("Do you want to remove noisy objects? (y/[n])", end='')
            answer = input()
            if answer.lower() == 'y':
                df = df.drop(noisyObjects)

    print("Do you want show pca? (y/[n])", end='')
    answer = input()
    if answer.lower() == 'y':
        show_pca(df)

    print("Do you want show t-SNE? (y/[n])", end='')
    answer = input()
    if answer.lower() == 'y':
        show_tsne(df)

    print("Do you want quit? (y/[n])", end='')
    answer = input()
    if answer.lower() == 'y':
        break
