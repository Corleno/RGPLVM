import pods
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import time
import pickle

if __name__ == "__main__":
    name = "Anuran_Genus"
    # Load Anuran Calls (MFCCs)
    data = pd.read_csv("../data/Frogs_MFCCs.csv")
    print(data.columns)
    Y = data.iloc[:,:22].values
    # labels = data["Family"].values
    labels = data["Genus"].values
    # labels = data["Species"].values
    unique_labels = np.unique(labels)
    label2num = {label:i for i, label in enumerate(unique_labels)}
    # import pdb
    # pdb.set_trace()
    label_nums = np.array([label2num[label] for label in labels])

    print(Y.shape, labels.shape)
    # (7195, 22) (7195,)
    print("Dimension: {}, # labels: {}".format(Y.shape[1], len(unique_labels)))
    # Dimension: 22, # labels: 8
    # import pdb
    # pdb.set_trace()

    # create dataframe
    df = pd.DataFrame(Y, columns=['feature' + str(i) for i in range(Y.shape[1])])
    df['label'] = labels
    df['label_num'] = label_nums
    # Visualize raw data using PCA and t-SNE
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(Y)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    # Explained variation per principal component: [0.38911606 0.21242301 0.10077414]
    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="label",
        palette=sns.color_palette("hls", 8),
        data=df.loc[rndperm, :],
        legend="full",
        alpha=0.3
    )
    plt.savefig("../res/anuran/pca_anuran.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 10)).gca(projection='3d')
    fig.scatter(xs=df.loc[rndperm, :]["pca-one"], ys=df.loc[rndperm, :]["pca-two"], zs=df.loc[rndperm, :]["pca-three"], c=df.loc[rndperm, :]["label_num"], cmap='tab10')
    fig.set_xlabel('pca-one')
    fig.set_ylabel('pca-two')
    fig.set_zlabel('pca-three')
    plt.savefig("../res/anuran/pca3d_anuran.png")


    # TSNE learning
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(Y)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", 8),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig("../res/anuran/tsne_anuran.png")
    plt.close(fig)

    # save low dimension embedding representation for both PCA and t-sne.
    with open("../res/anuran/pca_tsne_anuran.pickle", "wb") as res:
        pickle.dump(df, res)
