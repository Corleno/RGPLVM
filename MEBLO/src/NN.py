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
    # print(data.shape, D.shape)
    pred = np.asarray([label[np.argmin(D[n,:][D[n,:]>0]) if np.argmin(D[n,:][D[n,:]>0]) < n else np.argmin(D[n,:][D[n,:]>0])+1] for n in range(N)])
    pred_score = np.mean(pred == label)
    # knn = KNeighborsClassifier(n_neighbors = 3)
    # knn.fit(data, label)
    # pred_score = knn.score(data, label)
    if verbose:
        print("predictive accuracy: {}".format(pred_score))
    return pred_score

def cross_validation(data, label, n_cv = 10, verbose = False):
    N = data.shape[0]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_cv)
    kf.get_n_splits(data)
    pred_score = 0
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        pred_score += knn.score(X_test, y_test)
    return pred_score/n_cv

if __name__ == "__main__":
    pods.datasets.overide_manual_authorize = True  # dont ask to authorize
    np.random.seed(42)

    Q = 5
    M = 20

    # # Install the oil data
    # name = "oils"
    # data = pods.datasets.oil()
    # Y = data['X']
    # print('Number of points X Number of dimensions', Y.shape)
    # print(data['citation'])
    # labels = data['Y'].argmax(axis=1)

    #  # Install decampos_digits
    # name = "decampos_digits"
    # data = pods.datasets.decampos_digits()
    # Y = data['Y']
    # labels = data['lbls']

    # Load Anuran Calls (MFCCs)
    name = "Anuran_Genus"
    # Load Anuran Calls (MFCCs)
    data = pd.read_csv("../data/Frogs_MFCCs.csv")
    print(data.columns)
    Y = data.iloc[:,:22].values
    # labels = data["Family"].values
    labels = data["Genus"].values
    # labels = data["Species"].values
    print(Y.shape, labels.shape)


    # # Load result from {} bgplvm
    # with open("../res/{}_bgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
    #     data = pickle.load(res)
    # print("BGLVM, ACC: {}".format(NN_experiments(data, labels)))

    # Load result from name's vbgplvm
    with open("../res/{}_vbgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
        data = pickle.load(res)
    print("VBGLVM, ACC: {}".format(cross_validation(data, labels)))

    # # Load result from oil rbgplvm
    # with open("../res/{}_rbgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
    #     data = pickle.load(res)
    # print("RBGLVM, ACC: {}".format(NN_experiments(data, labels)))

    
    # Load result from name's rvbgplvm
    lamb = 20
    with open("../res/{}_rvbgplvm_Q{}_M{}_LAM{}.pickle".format(name, Q, M, lamb), "rb") as res:
        data = pickle.load(res)
    print("RVBGLVM, ACC: {}".format(cross_validation(data, labels)))

    lamb = 50
    with open("../res/{}_rvbgplvm_Q{}_M{}_LAM{}.pickle".format(name, Q, M, lamb), "rb") as res:
        data = pickle.load(res)
    print("RVBGLVM, ACC: {}".format(cross_validation(data, labels)))

    lamb = 100
    with open("../res/{}_rvbgplvm_Q{}_M{}_LAM{}.pickle".format(name, Q, M, lamb), "rb") as res:
        data = pickle.load(res)
    print("RVBGLVM, ACC: {}".format(cross_validation(data, labels)))
