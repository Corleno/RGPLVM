#!/user/bin/env python3
'''
Create  05212019

@author: meng2
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    reg = 0.0
    d = 0
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()   
    plt.title(r'$\lambda = 0, d = 0$') 
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)
    d = 1
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()    
    plt.title(r'$\lambda = 0, d = 1$')
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)


    reg = 20.0
    d = 0
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()    
    plt.title(r'$\lambda = 20, d = 0$')
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)
    d = 1
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()    
    plt.title(r'$\lambda = 20, d = 1$')
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)


    reg = 50.0
    d = 0
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()    
    plt.title(r'$\lambda = 50, d = 0$')
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)
    d = 1
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()    
    plt.title(r'$\lambda = 50, d = 1$')
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)


    reg = 100.0
    d = 0
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()    
    plt.title(r'$\lambda = 100, d = 0$')
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)
    d = 1
    with open("../res/latent_space_SIM_LAM{}_D{}.pickle".format(reg, d), "rb") as res:
        ms_2, IPs, labels, colors, sens_order = pickle.load(res)
    fig = plt.figure()
    for i, c in zip(np.unique(labels), colors):
        plt.scatter(ms_2[labels==i,sens_order[d][-1]], ms_2[labels==i,sens_order[d][-2]], color = c, label = "{},{}".format(int(i/3),i%3))
        plt.scatter(IPs[:,sens_order[d][-1]], IPs[:,sens_order[d][-2]], marker='x', label = "PI")
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    # plt.legend()    
    plt.title(r'$\lambda = 100, d = 1$')
    plt.savefig("../res/latent_space_SIM_LAM{}_D{}.png".format(reg, d))
    plt.close(fig)