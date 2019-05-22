#!/user/bin/env python3
'''
Create, 05/15/2019

@author: Rui Meng
'''

import numpy as np
import pandas as pd

import time
import sys
sys.path.append("..")

from GPflow import gpflow
from GPflow.gpflow.likelihoods import Gaussian
from GPflow.gpflow import kernels
from GPflow.gpflow.models.sgpr import SGPR, GPRFITC
from GPflow.gpflow.models.svgp import SVGP
from GPflow.gpflow.models.gpr import GPR
from GPflow.gpflow.training import AdamOptimizer, ScipyOptimizer, NatGradOptimizer

from scipy.stats import norm
from scipy.cluster.vq import kmeans2

import matplotlib.pyplot as plt 
from Utils.S_Plot import *



def batch_assess(model, assess_model, X, Y):
    n_batches = max(int(X.shape[0]/1000.), 1)
    lik, sq_diff, ms, vs = [], [], [], []
    for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
        l, sq, m, v = assess_model(model, X_batch, Y_batch)
        lik.append(l)
        sq_diff.append(sq)
        ms.append(m)
        vs.append(v)
    lik = np.concatenate(lik, 0)
    sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
    ms = np.concatenate(ms, 0)
    vs = np.concatenate(vs, 0)
    return np.average(lik), np.average(sq_diff)**0.5, ms, vs

def assess_single_layer(model, X_batch, Y_batch):
    m, v = model.predict_y(X_batch)
    lik = np.sum(norm.logpdf(Y_batch, loc=m, scale=v**0.5),  1)
    sq_diff = (m - Y_batch)**2
    return lik, sq_diff, m, v

if __name__ == "__main__":
    np.random.seed(22)
    n_input = 100
    X = np.random.rand(n_input,1)
    Y = np.sin(2 * X) + 0.2 * np.cos(22 * X) + np.random.randn(n_input, 1) * 0.01
    Xs = np.linspace(0, 1, 100)[:,None]
    Ys = np.sin(2 * Xs) + 0.2 * np.cos(22 * Xs) + np.random.randn(n_input, 1) * 0.01

    k = kernels.Matern32(1) + kernels.Linear(1)

    # # Gaussian Process Regression
    # name = "1D_sim_GPR"
    # m = GPR(X, Y, kern = k)
    # m.likelihood.variance = 0.01
    # # print(m.as_pandas_table())

    # # Optimization
    # m.compile()
    # opt= ScipyOptimizer()
    # opt.minimize(m, disp = False)
    # # print(m.as_pandas_table())

    # # Prediction
    # lik, rmse, ms, vs = batch_assess(m, assess_single_layer, Xs, Ys)
    # print("lik: {:.4f}, rmse:{:.4f}".format(lik,rmse))
    # # lik: 2.8355, rmse:0.0137

    # # SVGP
    # name = "1D_sim_SVGP"
    # Z_10 = kmeans2(X, 10, minit='points')[0]
    # print ("Z_10: {}".format(Z_10))

    # m = SVGP(X, Y, k, Gaussian(), Z_10)
    # m.likelihood.variance = 0.01
    # # print(m.as_pandas_table())

    # # Optimization
    # m.compile()
    # opt= ScipyOptimizer()
    # opt.minimize(m, disp = False)
    # # print(m.as_pandas_table())

    # # Prediction
    # lik, rmse, ms, vs = batch_assess(m, assess_single_layer, Xs, Ys)
    # print("lik: {:.4f}, rmse:{:.4f}".format(lik,rmse))
    # # lik: 2.2809, rmse:0.0163

    # # Plot
    # low_CI = (ms - 1.96*vs**0.5).reshape(-1)
    # mid_CI = ms.reshape(-1)
    # upper_CI = (ms + 1.96*vs**0.5).reshape(-1)
    # sorted_x = Xs.reshape(-1)
    # x_label = " "
    # y_label = " "
    # title = None
    # save_dir="../res/"
    # save_name= name + ".png"
    # Z = m.feature.Z.read_value().reshape(-1)
    # print(type(Z), Z)
    # lineplotCI(Xs, Ys, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title, save_dir = save_dir, save_name = save_name, Z = Z)

    # # SVGP_Fixed_0.5
    # name = "1D_sim_SVGP_Fixed_Z0"
    # Z_10 = np.linspace(0.25, 0.75, 10)[:,None]
    # print ("Z_10: {}".format(Z_10))

    # m = SVGP(X, Y, k, Gaussian(), Z_10)
    # m.likelihood.variance = 0.01
    # # print(m.as_pandas_table())

    # # Optimization
    # m.compile()
    # opt= ScipyOptimizer()
    # opt.minimize(m, disp = False)
    # # print(m.as_pandas_table())

    # # Prediction
    # lik, rmse, ms, vs = batch_assess(m, assess_single_layer, Xs, Ys)
    # print("lik: {:.4f}, rmse:{:.4f}".format(lik,rmse))
    # # lik: 0.5226, rmse:0.1424

    # # Plot
    # low_CI = (ms - 1.96*vs**0.5).reshape(-1)
    # mid_CI = ms.reshape(-1)
    # upper_CI = (ms + 1.96*vs**0.5).reshape(-1)
    # sorted_x = Xs.reshape(-1)
    # x_label = " "
    # y_label = " "
    # title = None
    # save_dir="../res/"
    # save_name= name + ".png"
    # Z = m.feature.Z.read_value().reshape(-1)
    # print(type(Z), Z)
    # lineplotCI(Xs, Ys, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title, save_dir = save_dir, save_name = save_name, Z = Z)


    # # SVGP_Fixed_1
    # name = "1D_sim_SVGP_Fixed_Z1"
    # Z_10 = np.linspace(0, 1.0, 10)[:,None]
    # print ("Z_10: {}".format(Z_10))

    # m = SVGP(X, Y, k, Gaussian(), Z_10)
    # m.likelihood.variance = 0.01
    # # print(m.as_pandas_table())

    # # Optimization
    # m.compile()
    # opt= ScipyOptimizer()
    # opt.minimize(m, disp = False)
    # # print(m.as_pandas_table())

    # # Prediction
    # lik, rmse, ms, vs = batch_assess(m, assess_single_layer, Xs, Ys)
    # print("lik: {:.4f}, rmse:{:.4f}".format(lik,rmse))
    # # lik: 1.7514, rmse:0.0383

    # # Plot
    # low_CI = (ms - 1.96*vs**0.5).reshape(-1)
    # mid_CI = ms.reshape(-1)
    # upper_CI = (ms + 1.96*vs**0.5).reshape(-1)
    # sorted_x = Xs.reshape(-1)
    # x_label = " "
    # y_label = " "
    # title = None
    # save_dir="../res/"
    # save_name= name + ".png"
    # Z = m.feature.Z.read_value().reshape(-1)
    # print(type(Z), Z)
    # lineplotCI(Xs, Ys, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title, save_dir = save_dir, save_name = save_name, Z = Z)

    # SVGP_Fixed_2
    name = "1D_sim_SVGP_Fixed_Z2"
    Z_10 = np.linspace(-0.5, 1.5, 10)[:,None]
    print ("Z_10: {}".format(Z_10))

    m = SVGP(X, Y, k, Gaussian(), Z_10)
    m.likelihood.variance = 0.01
    # print(m.as_pandas_table())

    # Optimization
    m.compile()
    opt= ScipyOptimizer()
    opt.minimize(m, disp = False)
    # print(m.as_pandas_table())

    # Prediction
    lik, rmse, ms, vs = batch_assess(m, assess_single_layer, Xs, Ys)
    print("lik: {:.4f}, rmse:{:.4f}".format(lik,rmse))
    # lik: 0.5400, rmse:0.1406

    # Plot
    low_CI = (ms - 1.96*vs**0.5).reshape(-1)
    mid_CI = ms.reshape(-1)
    upper_CI = (ms + 1.96*vs**0.5).reshape(-1)
    sorted_x = Xs.reshape(-1)
    x_label = " "
    y_label = " "
    title = None
    save_dir="../res/"
    save_name= name + ".png"
    Z = m.feature.Z.read_value().reshape(-1)
    print(type(Z), Z)
    lineplotCI(Xs, Ys, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title, save_dir = save_dir, save_name = save_name, Z = Z)

    # # SVGP_Fixed_5
    # name = "1D_sim_SVGP_Fixed_Z5"
    # Z_10 = np.linspace(0.5-2.5, 0.5+2.5, 10)[:,None]
    # print ("Z_10: {}".format(Z_10))

    # m = SVGP(X, Y, k, Gaussian(), Z_10)
    # m.likelihood.variance = 0.01
    # # print(m.as_pandas_table())

    # # Optimization
    # m.compile()
    # opt= ScipyOptimizer()
    # opt.minimize(m, disp = False)
    # # print(m.as_pandas_table())

    # # Prediction
    # lik, rmse, ms, vs = batch_assess(m, assess_single_layer, Xs, Ys)
    # print("lik: {:.4f}, rmse:{:.4f}".format(lik,rmse))
    # # lik: 0.4976, rmse:0.1463

    # # Plot
    # low_CI = (ms - 1.96*vs**0.5).reshape(-1)
    # mid_CI = ms.reshape(-1)
    # upper_CI = (ms + 1.96*vs**0.5).reshape(-1)
    # sorted_x = Xs.reshape(-1)
    # x_label = " "
    # y_label = " "
    # title = None
    # save_dir="../res/"
    # save_name= name + ".png"
    # Z = m.feature.Z.read_value().reshape(-1)
    # print(type(Z), Z)
    # lineplotCI(Xs, Ys, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title, save_dir = save_dir, save_name = save_name, Z = Z)
    