#!/user/bin/env python3
'''
Create  05172019

@author: meng2
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

from TCLGP import *

### Parallel Library
from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser=argparse.ArgumentParser()
    parser.add_argument("--M", help="number of inducing points", type=int, default=20)
    parser.add_argument("--Q", help="latent dimension size", type=int, default=5)
    parser.add_argument("--MC_T", help="Monte Carlo size", type=int, default=5)
    parser.add_argument("--method", help="methods: Adam, RSM, Adagrad", type=str, default='Adam')
    parser.add_argument("--dataset", help="name of dataset", type=str, default='sim_Markov_test_train')
    parser.add_argument("--reg", help="regularization", type=float, default=0.)
    parser.add_argument("--nugget", help="nugget effects in GPs", action='store_true')
    parser.add_argument("--training_epochs", help="number of training epochs", type=int, default=2000)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.1)
    parser.add_argument("--lower_bound", help="lower_bound of length scale in GP across time", type=float, default=0)
    parser.add_argument("--upper_bound", help="upper_bound of variance of nugget effects", type=float, default=0)
    parser.set_defaults(nugget=False)
    args=parser.parse_args()

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    # Load data
    with open('../data/'+args.dataset+'.dat', 'rb') as f:
        data_raw = pickle.load(f)
    T_train, T_test, Y_train, Y_test = data_raw
    ## T_train: N x T_train
    ## T_test: N
    ## Y_train: N x D x T_train
    ## Y_test: N x D 
    # Reset shape 
    Y_train = np.transpose(Y_train, axes = (0,2,1))
    N, T_max, D = Y_train.shape
    T_train = T_train.astype(np.float32)
    T_num_train = (np.ones(N)*T_max).astype(int)
    print(Y_train.shape, T_train.shape, T_num_train.shape)
    print(type(Y_train), type(T_train), type(T_num_train))
    K = [2,3]

    # Train data
    clgp_t = CLGP_T(M=args.M, Q=args.Q, MC_T=args.MC_T, reg=args.reg, nugget=args.nugget, error=1e-5, method=args.method, learning_rate=args.learning_rate, lower_bound=args.lower_bound, upper_bound=args.upper_bound)
    clgp_t.fit(Y_train, T_train, T_num_train, lamb = 0.001, learning_rate=args.learning_rate, training_epochs = args.training_epochs, method=args.method, verbose=False)

    # # predict training data
    # print("predict training data")
    # clgp_t.predict_categorical(clgp_t.est_m[:,-1,:])
    # # print("predicted result:{}".format(clgp_t.est_y_test))
    # # print("true result:{}".format(Y_train[:,-1,:]))
    # print("accuracy:{}".format(np.mean(clgp_t.est_y_test==Y_train[:,-1,:])))
    
    # predict test data
    print("predict testing data")
    clgp_t.predict_latent(T_test)
    clgp_t.predict_categorical()
    # print("predicted result:{}".format(clgp_t.est_y_test))
    # print("true result:{}".format(Y_test))
    pred_acc = np.mean(clgp_t.est_y_test==Y_test)
    print("accuracy:{}".format(pred_acc))
    
    pred_acc_list = comm.gather(pred_acc, root=0)
    if rank == 0:
    	np.savetxt("../res/pred_acc_LAM{}_Q{}".format(args.reg, args.Q), np.asarray(pred_acc_list))

    # predict individual 0 latent process
    # clgp_t.predict_latent_process(0, verbose=True)
    
    # # plot latent space
    # IPs = clgp_t.est_Z
    # ms = clgp_t.est_m
    # ms_2 = ms.reshape([-1, Q])

    # labels = Y_train.reshape([-1, D])
    # labels = (labels[:,0] * K[1] + labels[:, 1]).reshape(-1)
    # colors = cm.rainbow(np.linspace(0,1,K[0]*K[1]))

    # fig = plt.figure()
    # for i, c in zip(np.unique(labels), colors):
    #     plt.scatter(ms_2[labels==i,0], ms_2[labels==i,1], color = c, label = "{},{}".format(int(i/3),i%3))
    #     plt.scatter(IPs[:,0], IPs[:,1], marker='x', label = "PI")
    # # plt.legend()    
    # plt.savefig("../res/latent_space_SIM_LAM{}.png".format(args.reg))
    # plt.close(fig)



        