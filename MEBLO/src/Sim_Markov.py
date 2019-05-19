#!/usr/bin/env
#
# Rui Meng
#
# 05/17/2019

# Simulate categorical time series from Markov Model
import argparse
import numpy as np
import pickle

if __name__ == "__main__":
    np.random.seed(22)
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="the number of time series", type=int, default=100)
    parser.add_argument("--L", help="the length of time series", type=int, default=10)
    args = parser.parse_args()
    
    N = args.N
    L = args.L
    D = 2
    K = [2,3]

    # Set transition matrices
    T0 = np.array([[0.9, 0.1], [0.1, 0.9]])
    T1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    Tran = [T0, T1]

    T = np.tile(np.linspace(0,1,L), reps=(N,1))
    Y = np.zeros([N,D,L]).astype(int)
    for n in range(N):
        for d in range(D):
            Y[n,d,0] = np.random.choice(K[d])
            for t in range(L-1):
                Y[n,d,t+1] = np.random.choice(K[d], p = Tran[d][Y[n,d,t]])

    # with open("../data/sim_Markov.dat", "wb") as res:
    #     pickle.dump([T, Y], res)

    ### split to train and test
    T_train = T[:,:-1]
    T_test = T[:,-1]
    Y_train = Y[:,:,:-1]
    Y_test = Y[:,:,-1]
    train_test = [T_train, T_test, Y_train, Y_test]

    with open('../data/sim_Markov_test_train.dat', 'wb') as res:
        pickle.dump(train_test, res)

