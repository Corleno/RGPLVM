#!usr/bin/env python3
# author: Rui Meng
# date: 05132019

from __future__ import print_function
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pods
import pickle
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform
import pandas as pd

def NN_experiments(data, label, verbose = False):
    N = data.shape[0]
    D = squareform(pdist(data))
    print(data.shape, D.shape)
    pred = np.asarray([label[np.argmin(D[n,:][D[n,:]>0]) if np.argmin(D[n,:][D[n,:]>0]) < n else np.argmin(D[n,:][D[n,:]>0])+1] for n in range(N)])
    pred_score = np.mean(pred == label)
    # knn = KNeighborsClassifier(n_neighbors = 3)
    # knn.fit(data, label)
    # pred_score = knn.score(data, label)
    if verbose:
        print("predictive accuracy: {}".format(pred_score))
    return pred_score

if __name__ == "__main__":
    pods.datasets.overide_manual_authorize = True  # dont ask to authorize
    np.random.seed(42)

    Q = 10
    M = 20

    # Install the oil data
    name = "oils"
    data = pods.datasets.oil()
    Y = data['X']
    print('Number of points X Number of dimensions', Y.shape)
    print(data['citation'])
    labels = data['Y'].argmax(axis=1)

    #  # Install decampos_digits
    # name = "decampos_digits"
    # data = pods.datasets.decampos_digits()
    # Y = data['Y']
    # labels = data['lbls']

    # # Load Anuran Calls (MFCCs)
    # name = "Anuran_Genus"
    # # Load Anuran Calls (MFCCs)
    # data = pd.read_csv("../data/Frogs_MFCCs.csv")
    # print(data.columns)
    # Y = data.iloc[:,:22].values
    # # labels = data["Family"].values
    # labels = data["Genus"].values
    # # labels = data["Species"].values
    # print(Y.shape, labels.shape)


    # # Load result from {} bgplvm
    # with open("../res/{}_bgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
    #     data = pickle.load(res)
    # print("BGLVM, ACC: {}".format(NN_experiments(data, labels)))

    # Load result from oil vbgplvm
    with open("../res/{}_vbgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
        data = pickle.load(res)
    print("VBGLVM, ACC: {}".format(NN_experiments(data, labels)))

    # # Load result from oil rbgplvm
    # with open("../res/{}_rbgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
    #     data = pickle.load(res)
    # print("RBGLVM, ACC: {}".format(NN_experiments(data, labels)))

    # Load result from oil rvbgplvm
    with open("../res/{}_rvbgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
        data = pickle.load(res)
    print("RVBGLVM, ACC: {}".format(NN_experiments(data, labels)))
