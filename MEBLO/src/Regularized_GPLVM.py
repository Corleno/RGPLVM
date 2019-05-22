#!usr/bin/env python3
# author: Rui Meng
# date: 05072019

from __future__ import print_function
# import gpflow
# from gpflow import kernels
import sys
sys.path.append("..")
from GPflow import gpflow
from GPflow.gpflow import kernels
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
# %matplotlib inline
import pods
import pickle
import pandas as pd

def GPLVM_model(name):
    Q = 5
    M = 20
    N = Y.shape[0]
    X_mean = gpflow.models.PCA_reduce(Y, Q) # Initialise via PCA
    Z = np.random.permutation(X_mean.copy())[:M]

    fHmmm = False
    if(fHmmm):
        k = (kernels.RBF(3, ARD=True, active_dims=slice(0,3)) +
             kernels.Linear(2, ARD=False, active_dims=slice(3,5)))
    else:
        k = (kernels.RBF(3, ARD=True, active_dims=[0,1,2]) +
             kernels.Linear(2, ARD=False, active_dims=[3, 4]))

    # GPLVM
    GPLVM = gpflow.models.GPLVM(Y=Y, latent_dim=Q, X_mean=X_mean, kern=k)

    opt = gpflow.train.ScipyOptimizer()
    GPLVM.compile()
    opt.minimize(GPLVM)#, options=dict(disp=True, maxiter=100))

    # Compute and sensitivity to input
    # print(m.kern.kernels)
    kern = GPLVM.kern.kernels[0]
    sens = np.sqrt(kern.variance.read_value())/kern.lengthscales.read_value()
    print(GPLVM.kern)
    print(sens)
    # fig, ax = plt.subplots()
    # ax.bar(np.arange(len(kern.lengthscales.read_value())) , sens, 0.1, color='y')
    # ax.set_title('Sensitivity to latent inputs')
    # plt.savefig("../res/oils_sen.png")
    # plt.close(fig)

    return GPLVM, sens

def BGPLVM_model(name, Q = 10, M = 20):
    np.random.seed(22)
    # Create Bayesian GPLVM model using additive kernel
    # Q = 5
    # M = 20 # number of inducing pts
    N = Y.shape[0]
    X_mean = gpflow.models.PCA_reduce(Y, Q) # Initialise via PCA
    # X_mean = np.random.normal(size = [N, Q])
    Z = np.random.permutation(X_mean.copy())[:M]

    fHmmm = False
    if(fHmmm):
        k = (kernels.RBF(3, ARD=True, active_dims=slice(0,3)) +
             kernels.Linear(2, ARD=False, active_dims=slice(3,5)))
    else:
        # k = (kernels.RBF(3, ARD=True, active_dims=[0,1,2]) +
        #      kernels.Linear(2, ARD=False, active_dims=[3, 4]))
        k = kernels.RBF(Q, ARD=True)

    # Bayesian GPLVM
    BGPLVM = gpflow.models.BayesianGPLVM(X_mean=X_mean, X_var=0.1*np.ones((N, Q)), Y=Y,
                                kern=k, M=M, Z=Z)
    BGPLVM.likelihood.variance = 0.01

    opt = gpflow.train.ScipyOptimizer()
    BGPLVM.compile()
    opt.minimize(BGPLVM, disp = False, maxiter = 1000)#, options=dict(disp=True, maxiter=100))
    # print("###############################")
    # print(BGPLVM.X_mean.read_value(), BGPLVM.X_var.read_value())
    # print("###############################")

    # Compute and sensitivity to input
    # print(m.kern.kernels)
    kern = BGPLVM.kern
    sens = np.sqrt(kern.variance.read_value())/kern.lengthscales.read_value()
    print(BGPLVM.kern)
    print(sens)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(kern.lengthscales.read_value())) , sens, 0.1, color='y')
    ax.set_title('Sensitivity to latent inputs')
    plt.savefig("../res/{}_sen_bgplvm_Q{}_M{}.png".format(name,Q,M))
    plt.close(fig)
    with open("../res/{}_bgplvm_Q{}_M{}.pickle".format(name,Q,M), "wb") as res:
        pickle.dump(BGPLVM.X_mean.read_value(), res)

    return BGPLVM, sens

def VBGPLVM_model(name, Q = 10, M = 20, verbose = False):
    np.random.seed(22)
    # Create Bayesian GPLVM model using additive kernel
    # Q = 5
    # M = 20 # number of inducing pts
    N, D = Y.shape
    X_mean = gpflow.models.PCA_reduce(Y, Q) # Initialise via PCA
    # X_mean = np.random.normal(size = [N, Q])
    Z = np.random.permutation(X_mean.copy())[:M]

    fHmmm = False
    if(fHmmm):
        k = (kernels.RBF(3, ARD=True, active_dims=slice(0,3)) +
             kernels.Linear(2, ARD=False, active_dims=slice(3,5)))
    else:
        # k = (kernels.RBF(3, ARD=True, active_dims=[0,1,2]) +
        #      kernels.Linear(2, ARD=False, active_dims=[3, 4]))
        k = kernels.RBF(Q, ARD=True)


    # print(X_mean.shape)
    # VBGPLVM
    VBGPLVM = gpflow.models.VBGPLVM(X_mean=X_mean, X_var=0.1*np.ones((N, Q)), U_mean = np.zeros((M, D)), U_var = 0.01*np.ones((M, D)), Y=Y,
                                kern=k, M=M, Z=Z)
    VBGPLVM.likelihood.variance = 0.01
    # print(VBGPLVM.X_mean, VBGPLVM.X_var)
    
    opt = gpflow.train.ScipyOptimizer()
    VBGPLVM.compile()
    # print(VBGPLVM.compute_log_likelihood())
    opt.minimize(VBGPLVM, disp = False, maxiter = 1000)#, options=dict(disp=True, maxiter=100))
    
    # print("############################")
    # print("X_mean: ", VBGPLVM.X_mean.read_value())
    # print("X_var: ", VBGPLVM.X_var.read_value())
    # print("############################")
    # print("U_mean: ", VBGPLVM.U_mean.read_value())
    # print("U_var: ", VBGPLVM.U_var.read_value())
    # print("############################")

    # Compute and sensitivity to input
    # print(m.kern.kernels)
    kern = VBGPLVM.kern
    sens = np.sqrt(kern.variance.read_value())/kern.lengthscales.read_value()
    print(VBGPLVM.kern)
    print(sens)

    if verbose:
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(kern.lengthscales.read_value())) , sens, 0.1, color='y')
        ax.set_title('Sensitivity to latent inputs')
        plt.savefig("../res/test/{}_sen_vbgplvm_Q{}_M{}.png".format(name, Q, M))
        plt.close(fig)
        
        colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
        sens_order = np.argsort(sens)
        fig, ax = plt.subplots()
        for i, c in zip(np.unique(labels), colors):
            ax.scatter(VBGPLVM.X_mean.read_value()[labels==i,sens_order[-1]], VBGPLVM.X_mean.read_value()[labels==i,sens_order[-2]], color=c, label=i)
            ax.set_title('VBGPLVM')
            ax.scatter(VBGPLVM.feature.Z.read_value()[:,sens_order[-1]], VBGPLVM.feature.Z.read_value()[:,sens_order[-2]], label = "IP", marker='x')
        plt.savefig("../res/test/VBGPLVM.png")
        loglik = VBGPLVM.compute_log_likelihood()
        np.savetxt("../res/test/VBGPLVM.csv", np.asarray([loglik]))
    with open("../res/{}_vbgplvm_Q{}_M{}.pickle".format(name, Q, M), "wb") as res:
        pickle.dump([VBGPLVM.X_mean.read_value(), VBGPLVM.feature.Z.read_value(), labels, colors, sens_order], res)

    return VBGPLVM, sens

def RBGPLVM_model(name, Q = 10, M = 20, lamb = 20):
    np.random.seed(22)
    # Create Regularzed GPLVM model using additive kernel
    # Q = 5
    # M = 20 # number of inducing pts
    N = Y.shape[0]
    X_mean = gpflow.models.PCA_reduce(Y, Q) # Initialise via PCA
    Z = np.random.permutation(X_mean.copy())[:M]

    fHmmm = False
    if(fHmmm):
        k = (kernels.RBF(3, ARD=True, active_dims=slice(0,3)) +
             kernels.Linear(2, ARD=False, active_dims=slice(3,5)))
    else:
        # k = (kernels.RBF(3, ARD=True, active_dims=[0,1,2]) +
        #      kernels.Linear(2, ARD=False, active_dims=[3, 4]))
        k = kernels.RBF(Q, ARD=True)

    # Regularized GPLVM
    RBGPLVM = gpflow.models.RegularizedGPLVM(X_mean=X_mean, X_var=0.1*np.ones((N, Q)), Y=Y,
                                kern=k, M=M, Z=Z, lamb = lamb)
    RBGPLVM.likelihood.variance = 0.01

    opt = gpflow.train.ScipyOptimizer()
    RBGPLVM.compile()
    opt.minimize(RBGPLVM)#, options=dict(disp=True, maxiter=100))
    # print(m.X_mean, m.X_var)

    # Compute and sensitivity to input
    # print(m.kern.kernels)
    kern = RBGPLVM.kern
    sens = np.sqrt(kern.variance.read_value())/kern.lengthscales.read_value()
    print(RBGPLVM.kern)
    print(sens)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(kern.lengthscales.read_value())) , sens, 0.1, color='y')
    ax.set_title('Sensitivity to latent inputs')
    plt.savefig("../res/{}_sen_rbgplvm_Q{}_M{}.png".format(name, Q, M))
    plt.close(fig)
    with open("../res/{}_rbgplvm_Q{}_M{}.pickle".format(name, Q, M), "wb") as res:
        pickle.dump(RBGPLVM.X_mean.read_value(), res)


    return RBGPLVM, sens

def RVBGPLVM_model(name, Q = 10, M = 20, lamb = 20, verbose = False):
    np.random.seed(22)
    # Create Regularzed GPLVM model using additive kernel
    # Q = 5
    # M = 20 # number of inducing pts
    N, D = Y.shape
    X_mean = gpflow.models.PCA_reduce(Y, Q) # Initialise via PCA
    Z = np.random.permutation(X_mean.copy())[:M]

    fHmmm = False
    if(fHmmm):
        k = (kernels.RBF(3, ARD=True, active_dims=slice(0,3)) +
             kernels.Linear(2, ARD=False, active_dims=slice(3,5)))
    else:
        # k = (kernels.RBF(3, ARD=True, active_dims=[0,1,2]) +
        #      kernels.Linear(2, ARD=False, active_dims=[3, 4]))
        k = kernels.RBF(Q, ARD=True)

    # Regularized GPLVM
    RVBGPLVM = gpflow.models.RegularizedVBGPLVM(X_mean=X_mean, X_var=0.1*np.ones((N, Q)), U_mean = np.zeros((M, D)), U_var = 0.01*np.ones((M, D)), Y=Y,
                                kern=k, M=M, Z=Z, lamb = lamb)
    RVBGPLVM.likelihood.variance = 0.01

    opt = gpflow.train.ScipyOptimizer()
    RVBGPLVM.compile()
    opt.minimize(RVBGPLVM, disp = False)#, options=dict(disp=True, maxiter=100))

    # print(m.X_mean, m.X_var)

    # Compute and sensitivity to input
    # print(m.kern.kernels)
    kern = RVBGPLVM.kern
    sens = np.sqrt(kern.variance.read_value())/kern.lengthscales.read_value()
    print(RVBGPLVM.kern)
    print(sens)
    if verbose:
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(kern.lengthscales.read_value())) , sens, 0.1, color='y')
        ax.set_title('Sensitivity to latent inputs')
        plt.savefig("../res/test/{}_sen_rvbgplvm_Q{}_M{}_LAM{}.png".format(name, Q, M, lamb))
        plt.close(fig)
        
        colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
        sens_order = np.argsort(sens)
        fig, ax = plt.subplots()
        for i, c in zip(np.unique(labels), colors):
            ax.scatter(RVBGPLVM.X_mean.read_value()[labels==i,sens_order[-1]], RVBGPLVM.X_mean.read_value()[labels==i,sens_order[-2]], color=c, label=i)
            ax.set_title('RVBGPLVM LAM = {}'.format(lamb))
            ax.scatter(RVBGPLVM.feature.Z.read_value()[:,sens_order[-1]], RVBGPLVM.feature.Z.read_value()[:,sens_order[-2]], label = "IP", marker='x')
        plt.savefig("../res/test/RVBGPLVM_LAM{}.png".format(lamb))
        loglik = RVBGPLVM.compute_log_likelihood()
        np.savetxt("../res/test/RVBGPLVM_LAM{}.csv".format(lamb), np.asarray([loglik]))
  
    with open("../res/{}_rvbgplvm_Q{}_M{}_LAM{}.pickle".format(name, Q, M, lamb), "wb") as res:
        pickle.dump([RVBGPLVM.X_mean.read_value(), RVBGPLVM.feature.Z.read_value(), labels, colors, sens_order], res)


    return RVBGPLVM, sens


if __name__ == "__main__":
    pods.datasets.overide_manual_authorize = True  # dont ask to authorize
    np.random.seed(42)
    gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations
    
    # # Install the oil data
    # name = "oils"
    # data = pods.datasets.oil()
    # Y = data['X']
    # print('Number of points X Number of dimensions', Y.shape)
    # print(data['citation'])
    # labels = data['Y'].argmax(axis=1)

    # # Install decampos_digits
    # name = "decampos_digits"
    # data = pods.datasets.decampos_digits()
    # Y = data['Y']
    # labels = data['lbls'].reshape(-1)

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


    # settings
    Q = 5
    M = 20

    # # Bayesian GPLVM
    # BGPLVM, BGPLVM_sens = BGPLVM_model(name)
    # BGPLVM_sens_order = np.argsort(BGPLVM_sens)

    # VBGPLVM
    VBGPLVM, VBGPLVM_sens = VBGPLVM_model(name, Q = Q, M = M, verbose = True)
    VBGPLVM_sens_order = np.argsort(VBGPLVM_sens)

    # # Regularized GPLVM
    # RBGPLVM, RBGPLVM_sens = RBGPLVM_model(name)
    # RBGPLVM_sens_order = np.argsort(RBGPLVM_sens)    

    # # Regularized VBGPLVM
    # RVBGPLVM_10, RVBGPLVM_10_sens = RVBGPLVM_model(name, Q = Q, M = M, lamb = 10, verbose = True)
    # RVBGPLVM_10_sens_order = np.argsort(RVBGPLVM_10_sens)    
    # Regularized VBGPLVM
    RVBGPLVM_20, RVBGPLVM_20_sens = RVBGPLVM_model(name, Q = Q, M = M, lamb = 20, verbose = True)
    RVBGPLVM_20_sens_order = np.argsort(RVBGPLVM_20_sens)    
    # Regularized VBGPLVM
    RVBGPLVM_50, RVBGPLVM_50_sens = RVBGPLVM_model(name, Q = Q, M = M, lamb = 50, verbose = True)
    RVBGPLVM_50_sens_order = np.argsort(RVBGPLVM_50_sens)    
    # Regularized VBGPLVM
    RVBGPLVM_100, RVBGPLVM_100_sens = RVBGPLVM_model(name, Q = Q, M = M, lamb = 100, verbose = True)
    RVBGPLVM_100_sens_order = np.argsort(RVBGPLVM_100_sens)    


    # # Plotting vs PCA
    # XPCAplot = gpflow.models.PCA_reduce(Y, 2)
    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,6))
    # colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))

    # for i, c in zip(np.unique(labels), colors):
    #     ax[0, 0].scatter(XPCAplot[labels==i,0], XPCAplot[labels==i,1], color=c, label=i)
    #     ax[0, 0].set_title('PCA')
    #     # ax[1].scatter(BGPLVM.X_mean.read_value()[labels==i,BGPLVM_sens_order[-1]], BGPLVM.X_mean.read_value()[labels==i,BGPLVM_sens_order[-2]], color=c, label=i)
    #     # ax[1].set_title('Bayesian GPLVM')
    #     ax[0, 1].scatter(VBGPLVM.X_mean.read_value()[labels==i,VBGPLVM_sens_order[-1]], VBGPLVM.X_mean.read_value()[labels==i,VBGPLVM_sens_order[-2]], color=c, label=i)
    #     ax[0, 1].set_title('VBGPLVM')
    #     # ax[2].scatter(RBGPLVM.X_mean.read_value()[labels==i,RBGPLVM_sens_order[-1]], RBGPLVM.X_mean.read_value()[labels==i,RBGPLVM_sens_order[-2]], color=c, label=i)
    #     # ax[2].set_title('RBGPLVM')
    #     ax[0, 2].scatter(RVBGPLVM_10.X_mean.read_value()[labels==i,RVBGPLVM_10_sens_order[-1]], RVBGPLVM_10.X_mean.read_value()[labels==i,RVBGPLVM_10_sens_order[-2]], color=c, label=i)
    #     ax[0, 2].set_title('RVBGPLVM LAM = 10')
    #     ax[1, 0].scatter(RVBGPLVM_20.X_mean.read_value()[labels==i,RVBGPLVM_20_sens_order[-1]], RVBGPLVM_20.X_mean.read_value()[labels==i,RVBGPLVM_20_sens_order[-2]], color=c, label=i)
    #     ax[1, 0].set_title('RVBGPLVM LAM = 20')
    #     ax[1, 1].scatter(RVBGPLVM_50.X_mean.read_value()[labels==i,RVBGPLVM_50_sens_order[-1]], RVBGPLVM_50.X_mean.read_value()[labels==i,RVBGPLVM_50_sens_order[-2]], color=c, label=i)
    #     ax[1, 1].set_title('RVBGPLVM LAM = 50')
    #     ax[1, 2].scatter(RVBGPLVM_100.X_mean.read_value()[labels==i,RVBGPLVM_100_sens_order[-1]], RVBGPLVM_100.X_mean.read_value()[labels==i,RVBGPLVM_100_sens_order[-2]], color=c, label=i)
    #     ax[1, 2].set_title('RVBGPLVM LAM = 100')
 

    # # Plot inducing points
    # # ax[1].scatter(BGPLVM.feature.Z.read_value()[:,BGPLVM_sens_order[-1]], BGPLVM.feature.Z.read_value()[:,BGPLVM_sens_order[-2]], label = "IP", marker='x')
    # ax[0, 1].scatter(VBGPLVM.feature.Z.read_value()[:,VBGPLVM_sens_order[-1]], VBGPLVM.feature.Z.read_value()[:,VBGPLVM_sens_order[-2]], label = "IP", marker='x')
    # # ax[2].scatter(RBGPLVM.feature.Z.read_value()[:,RBGPLVM_sens_order[-1]], RBGPLVM.feature.Z.read_value()[:,RBGPLVM_sens_order[-2]], label = "IP", marker='x')
    # ax[0, 2].scatter(RVBGPLVM_10.feature.Z.read_value()[:,RVBGPLVM_10_sens_order[-1]], RVBGPLVM_10.feature.Z.read_value()[:,RVBGPLVM_10_sens_order[-2]], label = "IP", marker='x')
    # ax[1, 0].scatter(RVBGPLVM_20.feature.Z.read_value()[:,RVBGPLVM_20_sens_order[-1]], RVBGPLVM_20.feature.Z.read_value()[:,RVBGPLVM_20_sens_order[-2]], label = "IP", marker='x')
    # ax[1, 1].scatter(RVBGPLVM_50.feature.Z.read_value()[:,RVBGPLVM_50_sens_order[-1]], RVBGPLVM_50.feature.Z.read_value()[:,RVBGPLVM_50_sens_order[-2]], label = "IP", marker='x')
    # ax[1, 2].scatter(RVBGPLVM_100.feature.Z.read_value()[:,RVBGPLVM_100_sens_order[-1]], RVBGPLVM_100.feature.Z.read_value()[:,RVBGPLVM_100_sens_order[-2]], label = "IP", marker='x')
    
    # plt.savefig("../res/{}_latent_Q{}_M{}.png".format(name,Q,M))
    # plt.close()
