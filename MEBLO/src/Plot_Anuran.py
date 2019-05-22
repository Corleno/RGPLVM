#!/user/bin/env python3
'''
Create  05212019

@author: meng2
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    name = "Anuran_Genus"
    Q = 5
    M = 20

    with open("../res/{}_vbgplvm_Q{}_M{}.pickle".format(name, Q, M), "rb") as res:
        X_mean, Z, labels, colors, sens_order = pickle.load(res)
    fig, ax = plt.subplots()
    for i, c in zip(np.unique(labels), colors):
        ax.scatter(X_mean[labels==i,sens_order[-1]], X_mean[labels==i,sens_order[-2]], color=c, label=i)
        ax.set_title(r'$\lambda = 0$')
        # ax.set_title('VBGPLVM')
        ax.scatter(Z[:,sens_order[-1]], Z[:,sens_order[-2]], label = "IP", marker='x')
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    plt.savefig("../res/test/VBGPLVM.png")

    # lamb = 20, 50, 100
    lamb = 20
    with open("../res/{}_rvbgplvm_Q{}_M{}_LAM{}.pickle".format(name, Q, M, lamb), "rb") as res:
        X_mean, Z, labels, colors, sens_order = pickle.load(res)
    fig, ax = plt.subplots()
    for i, c in zip(np.unique(labels), colors):
        ax.scatter(X_mean[labels==i,sens_order[-1]], X_mean[labels==i,sens_order[-2]], color=c, label=i)
        ax.set_title(r'$\lambda = 20$')
        # ax.set_title('RVBGPLVM LAM = {}'.format(lamb))
        ax.scatter(Z[:,sens_order[-1]], Z[:,sens_order[-2]], label = "IP", marker='x')
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)    
    plt.savefig("../res/test/RVBGPLVM_LAM{}.png".format(lamb))
    lamb = 50
    with open("../res/{}_rvbgplvm_Q{}_M{}_LAM{}.pickle".format(name, Q, M, lamb), "rb") as res:
        X_mean, Z, labels, colors, sens_order = pickle.load(res)
    fig, ax = plt.subplots()
    for i, c in zip(np.unique(labels), colors):
        ax.scatter(X_mean[labels==i,sens_order[-1]], X_mean[labels==i,sens_order[-2]], color=c, label=i)
        ax.set_title(r'$\lambda = 50$')
        # ax.set_title('RVBGPLVM LAM = {}'.format(lamb))
        ax.scatter(Z[:,sens_order[-1]], Z[:,sens_order[-2]], label = "IP", marker='x')
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)    
    plt.savefig("../res/test/RVBGPLVM_LAM{}.png".format(lamb))
    lamb = 100
    with open("../res/{}_rvbgplvm_Q{}_M{}_LAM{}.pickle".format(name, Q, M, lamb), "rb") as res:
        X_mean, Z, labels, colors, sens_order = pickle.load(res)
    fig, ax = plt.subplots()
    for i, c in zip(np.unique(labels), colors):
        ax.scatter(X_mean[labels==i,sens_order[-1]], X_mean[labels==i,sens_order[-2]], color=c, label=i)
        ax.set_title(r'$\lambda = 100$')
        # ax.set_title('RVBGPLVM LAM = {}'.format(lamb))
        ax.scatter(Z[:,sens_order[-1]], Z[:,sens_order[-2]], label = "IP", marker='x')
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)    
    plt.savefig("../res/test/RVBGPLVM_LAM{}.png".format(lamb))