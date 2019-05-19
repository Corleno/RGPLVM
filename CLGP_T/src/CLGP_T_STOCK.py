#!/user/bin/env python3
'''
Create , 2018

@author: meng2
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
import argparse
import logging
import os
import seaborn as sns
import tensorflow_probability as tfp


class CLGP_T():
    """
    Variation Inference for temporal categorical latent gaussian process model
    """

    def __init__(self, M=20, Q=2, MC_T=5, error=1e-3, reg=0, nugget=False):
    # super().__init__('CLGP_T')
        self.M=M
        self.Q=Q
        self.MC_T=MC_T
        self.error=error
        self.reg=reg
        self.nugget=nugget

    # Define variables
    def _L2Ld(self, L):
        idx = np.zeros([self.M, self.M], dtype=np.int32)
        mask = np.zeros([self.M, self.M], dtype=np.bool)
        triu_idx = np.triu_indices(self.M)
        idx[triu_idx] = np.arange((self.M*(self.M+1)/2))
        mask[triu_idx] = True
        Ld = tf.where(mask, tf.gather(L, idx), tf.zeros([self.M, self.M], dtype=L.dtype))
        return tf.transpose(Ld)

    def _Cov_mat(self, theta, X1, X2 = None, nugget=0):
        # theta = (alpha, Lambda)
        sigmaf2= theta[0]
        _X2 = X1 if X2 is None else X2
        if len(X1.shape) == 1:
            X1 = tf.reshape(X1, [-1, 1])
        if len(_X2.shape) == 1:
            _X2 = tf.reshape(_X2, [-1, 1])
        l = theta[(theta.shape[0]-X1.shape[1]):]
        dist = tf.matmul(tf.reshape(tf.reduce_sum((X1/l)**2,1), [-1,1]), tf.reshape(tf.ones(_X2.shape[0]), [1,-1])) + tf.matmul(tf.reshape(tf.ones(X1.shape[0]), [-1,1]), tf.reshape(tf.reduce_sum((_X2/l)**2,1), [1,-1])) - 2*tf.matmul((X1/l), tf.transpose(_X2/l))
        cov_mat = sigmaf2 * tf.exp(-dist/2.0)
        if X2 is None:
            # To guarantee the robustness of matrix inversion, we add jitters on the diagonal.
            cov_mat += np.diag(np.ones(X1.shape[0]))*self.error
            # e, v = tf.self_adjoint_eig(cov_mat)
            # e = tf.where(e > self.error, e, self.error*tf.ones_like(e))
            # cov_mat = tf.matmul(tf.matmul(v, tf.matrix_diag(e)),tf.transpose(v))
            if self.nugget:
                cov_mat += tf.matrix_diag(tf.ones(X1.shape[0])*nugget)
        return cov_mat

    def _tf_cov(self, x):
        mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        cov_xx += np.diag(np.ones(cov_xx.shape[0]))*self.error
        return cov_xx

    def _tranformation_f(self, vec, lower_bound=0, upper_bound=0.5):
        # this function is used to transform variables from R domain to [a,b] domain.float32
        # first, we transform vec to vec0 from R to [0,1] using logistic function
        vec0 = 1.0/(1.0 + tf.exp(-vec))
        return lower_bound+vec0*(upper_bound-lower_bound)

    def _create_graph(self):
        with tf.variable_scope('network_build') as net_build:
            self.lamb = tf.placeholder(tf.float32, name = 'lamb')
            self.ltheta = tf.Variable(tf.zeros([self.D, self.Q + 1], dtype=tf.float32), name="ltheta")
            if not np.isnan(args.phi_s2) and not np.isnan(args.phi_l):
                self.lphi = tf.constant(np.log([[args.phi_s2, args.phi_l],[args.phi_s2, args.phi_l]]), dtype=tf.float32)
            else:
                self.lphi = tf.Variable(tf.zeros([self.Q, 2]), dtype=tf.float32, name = "lphi")
            # self.lphi = tf.constant(np.log([[1, 0.1], [1, 0.1]]), dtype=tf.float32, name = "lphi")
            self.Z = tf.Variable(np.random.randn(self.M, self.Q), dtype=tf.float32, name="Z")
            self.mu = tf.Variable(np.random.randn(self.M, self.D, self.K_max)*0.01, dtype=tf.float32, name="mu")
            self.L = tf.Variable(np.random.randn(self.D, self.M*(self.M+1)//2), dtype=tf.float32, name="L")
            self.m = tf.Variable(tf.zeros([self.N, self.T_max, self.Q], dtype=tf.float32), name="m")
            self.s = tf.Variable(np.log(np.ones([self.N, self.T_max, self.Q])*0.1), dtype=tf.float32, name="s")
            if self.nugget:
                self.ltheta_nugget = tf.Variable(np.log(np.ones(self.D)*0.01), dtype=tf.float32, name="ltheta_nugget")
                self.lphi_nugget = tf.Variable(np.log(np.ones(self.Q)*0.01), dtype=tf.float32, name="lphi_nugget")
                if args.upper_bound !=0:
                    self.theta_nugget = self._tranformation_f(self.ltheta_nugget, upper_bound = args.upper_bound)
                    self.phi_nugget = self._tranformation_f(self.lphi_nugget, upper_bound = args.upper_bound)
                else:
                    self.theta_nugget = tf.exp(self.ltheta_nugget)
                    self.phi_nugget = tf.exp(self.lphi_nugget)
                            
            self.theta = tf.exp(self.ltheta) 

            # Optimize hyper-parameters of GP across time.
            # Convert log scale parameters to standard scale
            # simple transformation.
            # self.phi = tf.exp(self.lphi) 
            # advanced transoformation.
            if args.lower_bound != 0:
                self.phi = tf.stack([tf.exp(self.lphi[:,0]), self._tranformation_f(self.lphi[:,1], lower_bound=args.lower_bound)],axis=1)
            else:
                self.phi = tf.exp(self.lphi)

            # Fix hyper-parameters of GP across time.
            # self.phi = tf.constant(np.tile([[1,0.1]],[self.Q,1]), dtype=tf.float32)

        tf.set_random_seed(1234)
        tfd = tfp.distributions

        cov_mat_x_list = []
        for q in range(self.Q):
            for n in range(self.N):
                if self.nugget:
                    cov_mat_x_list.append(self._Cov_mat(self.phi[q,:], self.y_t[n, :self.T[n]], nugget=self.phi_nugget[q]))
                else:
                    cov_mat_x_list.append(self._Cov_mat(self.phi[q,:], self.y_t[n, :self.T[n]]))


        # with tf.name_scope('Dist_X'):
            # # variational distribution of X independent Gaussian with dimension Q by N by T_max.
            # QX_loc = [[self.m[n, :self.T[n], q] for n in range(self.N)] for q in range(self.Q)] # Q by N by T_max
            # QX_scale_diag = [[tf.exp(self.s)[n, :self.T[n], q] for n in range(self.N)] for q in range(self.Q)]
            # print(tfd.MultivariateNormalDiag(loc = QX_loc, scale_diag=QX_scale_diag))
            # QX = tfd.Independent(distribution=tfd.MultivariateNormalDiag(loc = QX_loc, scale_diag=QX_scale_diag), reinterpreted_batch_ndims=0, name = "QX")
            # import pdb
            # pdb.set_trace()
            # # prior distribution of X from Gauusian process prior with dimentsion Q by N by T
            # PX_loc = [[tf.zeros(self.T[n]) for n in range(self.N)] for q in range(self.Q)]
            # PX_covariance_matrix = [[cov_mat_x_list[q*self.N+n] for n in range(self.N)] for q in range(self.Q)]
            # PX = tfd.Independent(distribution=tfd.MultivariateNormalFullCovariance(loc = PX_loc, covariance_matrix= PX_covariance_matrix), reinterpreted_batch_ndims=0, name = "PX")
            # QX_loc = tf.concat([[self.m[n, :self.T[n], q] for n in range(self.N)] for q in range(self.Q)], axis = 0)
            # import pdb
            # pdb.set_trace()

        Ld_list = []
        cov_mat_list = []
        for d in range(self.D):
            Ld = self.L[d,:]
            Ld_list.append(self._L2Ld(Ld))
            if self.nugget:
                cov_mat_list.append(self._Cov_mat(self.theta[d,:], self.Z, nugget=self.theta_nugget[d]))
            else:
                cov_mat_list.append(self._Cov_mat(self.theta[d,:], self.Z))

        with tf.name_scope('Dist_U'):
            QU_loc = [[self.mu[:, d, k] for k in range(self.K_max)] for d in range(self.D)]
            eyes = tf.eye(self.M)
            QU_scale_tril = [[Ld_list[d] if k < self.K[d] else eyes for k in range(self.K_max)] for d in range(self.D)]
            QU = tfd.Independent(distribution = tfd.MultivariateNormalTriL(loc = QU_loc, scale_tril=QU_scale_tril), reinterpreted_batch_ndims=0, name = "QU")
            PU_loc = [[tf.zeros(self.M) for k in range(self.K_max)] for d in range(self.D)]
            PU_covariance_matrix = [[cov_mat_list[d] if k < self.K[d] else eyes for k in range(self.K_max)] for d in range(self.D)]
            PU = tfd.Independent(distribution= tfd.MultivariateNormalFullCovariance(loc = PU_loc, covariance_matrix= PU_covariance_matrix), reinterpreted_batch_ndims=0, name = "PU")

        with tf.name_scope('KL_X'):
            self.KL_X = 0
            for q in range(self.Q):
                for n in range(self.N):
                    QX_loc = self.m[n, :self.T[n], q]
                    QX_scale_diag = tf.exp(self.s)[n, :self.T[n], q]
                    QX = tfd.MultivariateNormalDiag(loc = QX_loc, scale_diag=QX_scale_diag)
                    PX_loc = tf.zeros(self.T[n])
                    PX_covariance_matrix = cov_mat_x_list[q*self.N+n]
                    PX = tfd.MultivariateNormalFullCovariance(loc = PX_loc, covariance_matrix= PX_covariance_matrix)
                    self.KL_X += tf.distributions.kl_divergence(QX, PX)
            # self.KL_X = tf.reduce_sum(tf.distributions.kl_divergence(QX, PX, name = 'KL_X'))
            tf.summary.scalar("summary/KL_X", self.KL_X)

        with tf.name_scope('KL_U'):
            KL_U_mat = tf.distributions.kl_divergence(QU, PU, name = 'KL_U_mat')
            indx = [[d, k] for d in range(self.D) for k in range(self.K[d])]
            self.KL_U = tf.reduce_sum(tf.gather_nd(KL_U_mat, indx), name = 'KL_U')
            tf.summary.scalar("summary/KL_U", self.KL_U)

        with tf.name_scope('KL_ZX'):
            # estimate distribution of all Zs using gaussian distribution
            Q_Z_loc = tf.reduce_mean(self.Z, axis=0)
            Q_Z_cov = self._tf_cov(self.Z)
            Q_Z = tfd.MultivariateNormalFullCovariance(loc=Q_Z_loc, covariance_matrix=Q_Z_cov, name='ED_Z')
            # estimate distribution of all Xs for each time using gaussian distribution
            m_mat = tf.concat([self.m[n, :self.T[n], :] for n in range(self.N)], axis =0)
            Q_X_loc = tf.reduce_mean(m_mat, axis=0)
            Q_X_cov = self._tf_cov(m_mat)
            Q_X = tfd.MultivariateNormalFullCovariance(loc=Q_X_loc, covariance_matrix=Q_X_cov, name='ED_X')
            # compute the KL divergence between X and Z
            self.KL_ZX = tf.distributions.kl_divergence(Q_Z, Q_X, name = 'KL_ZX')

        with tf.name_scope('Comp_F'):
            self.Comp_F = tf.constant(0, dtype=tf.float32, name="Comp_F")
            for tt in range(self.MC_T):
                Comp_F_tt = 0

                ### Sample X
                sampled_eps_X = tf.random_normal([self.N, self.T_max, self.Q])
                sampled_X = self.m + tf.multiply(tf.exp(self.s), sampled_eps_X)
                print ("QX has been sampled.")

                ### Sample U
                sampled_U = []
                for d in range(self.D):
                    sampled_eps_U_d = tf.random_normal([self.M, self.K[d]])
                    sampled_U_d = self.mu[:, d, :self.K[d]] + tf.matmul(Ld_list[d], sampled_eps_U_d)
                    paddings = np.array([[0, 0], [0, self.K_max-self.K[d]]])
                    sampled_U.append(tf.pad(sampled_U_d, paddings))
                print ("QU has been sampled.")
                sampled_U = tf.stack(sampled_U)

                ### Sample F and compute log-likelihood
                sampled_X_stacked = tf.concat([sampled_X[n, :self.T[n], :] for n in range(self.N)], axis=0)
                sampled_f =[]
                for d in range(self.D):
                    
                    if self.nugget:
                        Inv_cov_d_MM = tf.matrix_inverse(self._Cov_mat(self.theta[d,:], self.Z, nugget=self.theta_nugget[d]))
                    else:
                        Inv_cov_d_MM = tf.matrix_inverse(self._Cov_mat(self.theta[d,:], self.Z))
                    cov_NM = self._Cov_mat(self.theta[d,:], sampled_X_stacked, self.Z)
                    A_d = tf.matmul(cov_NM, Inv_cov_d_MM) #shape: Sum_T by M
                    B_d = tf.reshape(tf.reduce_sum(tf.multiply(A_d, cov_NM), axis=1), [-1]) #shape Sum_T
                    if self.nugget:
                            b_d = (self.theta[d,0] + self.theta_nugget[d])*tf.constant(np.ones(B_d.shape[0]), dtype=np.float32) - B_d
                    else:
                        b_d = (self.theta[d,0])*tf.constant(np.ones(B_d.shape[0]), dtype=np.float32) - B_d
                    zeros = tf.zeros_like(b_d)
                    masked = b_d > 0
                    b_d = tf.where(masked, b_d, zeros)

                    sampled_eps_f_d = tf.random_normal([sampled_X_stacked.shape.as_list()[0], self.K[d]])
                    sampled_f_d = tf.matmul(A_d, sampled_U[d,:,:self.K[d]]) + tf.multiply(tf.tile(tf.reshape(tf.sqrt(b_d), [-1,1]), [1,self.K[d]]), sampled_eps_f_d)
                    y_d_indx = np.stack([np.linspace(0, self.y_stacked.shape[0]-1, self.y_stacked.shape[0]), self.y_stacked[:,d]], axis = 1).astype(np.int32)
                    Comp_F_tt += tf.reduce_sum(tf.log(tf.gather_nd(tf.nn.softmax(sampled_f_d), y_d_indx)))
                    paddings = np.array([[0,0], [0,self.K_max-self.K[d]]])
                    sampled_f.append(tf.pad(sampled_f_d, paddings))
                sampled_f = tf.stack(sampled_f)

                self.Comp_F += Comp_F_tt
                print ("QF has been sampled.")
                print ("{}th Monte Carlo iteration has been completed.".format(tt))
            self.Comp_F = self.Comp_F/self.MC_T
            tf.summary.scalar("summary/Comp_F", self.Comp_F)

    def _create_loss_optimizer(self):
        with tf.name_scope('elbo'):
            if self.reg == 0:
                self.elbo = - self.lamb*(self.KL_X + self.KL_U) + self.Comp_F
            else:
                self.elbo = - self.lamb*(self.KL_X + self.KL_U + self.reg*self.KL_ZX) + self.Comp_F
            # self.elbo = - self.lamb*(self.KL_X + self.KL_U) + self.Comp_F
            negelbo = - self.elbo

        with tf.variable_scope('network_train') as net_train:
            with tf.name_scope('train'):
                if self.method=="Adam":
                    opt = tf.train.AdamOptimizer(self.learning_rate)
                if self.method=='RSM':
                    opt = tf.train.RMSPropOptimizer(self.learning_rate)
                if self.method=='Adagrad':
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                self.train = opt.minimize(negelbo)
                # self.train_phi = opt.minimize(self.KL_X, var_list=[self.lphi])
                # self.train_all_but_phi = opt.minimize(negelbo, var_list=[self.ltheta, self.Z, self.mu, self.L, self.m, self.s])

    def _create_summary(self):
        self.summ =  tf.summary.merge_all()

    def fit(self, Y_train, T_train, T_num_train, display_step=5, learning_rate=0.1, lamb=1, lamb_inc=0.001, training_epochs=2000, model_path='/tmp/model_tem.ckpt', hist_path='./logs/tem', method='Adam', verbose=False):
        self.learning_rate=learning_rate
        self.training_epochs=training_epochs
        self.method=method
        self.N, self.T_max, self.D = Y_train.shape
        self.y = Y_train
        self.y_t = T_train
        self.T = T_num_train
        self.y_stacked = np.concatenate([Y_train[n, :self.T[n], :]for n in range(self.N)], axis=0)

        var_levels = np.zeros(self.D, dtype=np.int)
        for d in range(self.D):
            var_levels[d] = len(np.unique(self.y_stacked[:, d].reshape([-1])))
        self.K = var_levels
        self.K_max = max(self.K)

        # create graph
        self._create_graph()
        # create optimizer
        self._create_loss_optimizer()
        # create summary
        self._create_summary()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

        ### create a history saver
        writer = tf.summary.FileWriter(hist_path, self.sess.graph)
        ### cretate a model saver
        saver = tf.train.Saver()

        elbo_hist=[]
        # Training with KL annealing
        for epoch in range(self.training_epochs):
            self.sess.run(self.train, feed_dict={self.lamb: lamb})
            # self.sess.run(self.train_all_but_phi, feed_dict={self.lamb: lamb})
            # self.sess.run(self.train_phi)
            if epoch % display_step == 0:
                # print training information
                lamb0, elbo0, theta0, phi0, Z0, KL_U0, KL_X0, KL_ZX0, Comp_F0, summary = self.sess.run([self.lamb, self.elbo, self.theta, self.phi, self.Z, self.KL_U, self.KL_X, self.KL_ZX, self.Comp_F, self.summ], feed_dict={self.lamb: lamb})
                print("Epoch: {}".format(epoch+1))
                print("elbo: {}, F: {}, KL_U:{}, KL_X:{}, KL_ZX:{}, phi:{}".format(elbo0, Comp_F0, KL_U0, KL_X0, KL_ZX0, phi0))
            writer.add_summary(summary, epoch)
            lamb = min(1, lamb_inc+lamb)
            elbo_hist.append(elbo0)

        if verbose:
            fig=plt.figure()
            plt.plot(elbo_hist)
            plt.title('Elbo_trace')
            fig.savefig('EBLO_trace_{}.png'.format(args.method))

        # Close history, saver, model saver and session
        writer.close()
        print("History has been saved under {}".format(hist_path))
        # Save model weights to disk
        saver.save(self.sess, model_path)
        print("Model has beend saved under{}".format(model_path))

        # Save estimated variational parameters
        if self.nugget:
            self.est_elbo, self.est_theta, self.est_Z, self.est_mu, self.est_L, self.est_m, self.est_s, self.est_phi, self.est_theta_nugget, self.est_phi_nugget = self.sess.run([self.elbo, self.theta, self.Z, self.mu, self.L, self.m, self.s, self.phi, self.theta_nugget, self.phi_nugget], feed_dict={self.lamb: 1})
        else:
            self.est_elbo, self.est_theta, self.est_Z, self.est_mu, self.est_L, self.est_m, self.est_s, self.est_phi = self.sess.run([self.elbo, self.theta, self.Z, self.mu, self.L, self.m, self.s, self.phi], feed_dict={self.lamb: 1})
        self.sess.close()

    def Cov_mat(self, theta, X1, X2 = None, nugget=0):
        # theta = variance + scale parameters. parameters are all under log scale.
        sigmaf2= theta[0]
        _X2 = X1 if X2 is None else X2
        # guarantee X1 is a two-dimensional array
        if len(X1.shape) == 0:
            X1 = np.array([[X1]])
        if len(X1.shape) == 1:
            X1 = np.reshape(X1, [-1, 1])
        if len(_X2.shape) == 0:
            _X2 = np.array([[_X2]])
        if len(_X2.shape) == 1:
            _X2 = np.reshape(_X2, [-1, 1])
        #l: Automatic relevance determination
        l = theta[(theta.shape[0]-X1.shape[1]):]
        dist = np.matmul(np.reshape(np.sum((X1/l)**2,1), [-1,1]), np.reshape(np.ones(_X2.shape[0]), [1,-1])) + np.matmul(np.reshape(np.ones(X1.shape[0]), [-1,1]), np.reshape(np.sum((_X2/l)**2,1), [1,-1])) - 2*np.matmul((X1/l), np.transpose(_X2/l))
        cov_mat = sigmaf2 * np.exp(-dist/2.0)
        if X2 is None:
            # To guarantee the robustness of matrix inversion, we add epsilion on the diagonal.
            cov_mat += np.diag(np.ones(X1.shape[0]))*self.error
            # e, v = np.linalg.eig(cov_mat)
            # e = np.where(e > self.error, e, self.error*np.ones_like(e))
            # cov_mat = np.matmul(np.matmul(v, np.diag(e)), v.T)
            if self.nugget:
                cov_mat += np.diag(np.ones(X1.shape[0])*nugget)
        return cov_mat

    def predict_latent(self, T_test):
        # Predict latent variables on T_test using posterior mean
        x_test =[]
        for q in range(self.Q):
            x_test_q = []
            for n in range(self.N):
                if self.nugget:
                    Cov_MM_inv = np.linalg.inv(self.Cov_mat(self.est_phi[q,:], self.y_t[n,:self.T[n]], nugget=self.est_phi_nugget[q]))
                else:
                    Cov_MM_inv = np.linalg.inv(self.Cov_mat(self.est_phi[q,:], self.y_t[n,:self.T[n]]))
                Cov_Mn = self.Cov_mat(self.est_phi[q,:], self.y_t[n,:self.T[n]], T_test[n])
                Cov_nM = Cov_Mn.T
                A = np.matmul(Cov_nM, Cov_MM_inv)
                x_test_q.append(np.matmul(A, self.est_m[n,:self.T[n],q].reshape([-1, 1]))[0,0])
            x_test_q = np.array(x_test_q)
            x_test.append(x_test_q)
        self.est_x_test = np.array(x_test).T # N by Q

    def predict_latent_process(self, patient_ind, T, n_grids=100, verbose=False):
        # Predict the latent process for individual patient
        T_all = np.concatenate([self.y_t[n, :self.T[n]] for n in range(self.N)], axis=0) 
        T_range = np.array([np.min(T_all), np.max(T_all)])
        T_len = np.max(T_all)-np.min(T_all)
        grids = np.linspace(T_range[0]-0.1*T_len, T_range[1]+0.1*T_len, num=n_grids)
        quan = np.zeros([self.Q,3,n_grids])
        for q in range(self.Q):
            #print('Cov_MM: ', self.Cov_mat(self.est_phi[q,:], self.y_t[patient_ind,:self.T[patient_ind]]))
            for i in range(n_grids):
                if self.nugget:
                    Cov_MM_inv = np.linalg.inv(self.Cov_mat(self.est_phi[q,:], self.y_t[patient_ind,:self.T[patient_ind]], nugget=self.est_phi_nugget[q]))
                else:
                    Cov_MM_inv = np.linalg.inv(self.Cov_mat(self.est_phi[q,:], self.y_t[patient_ind,:self.T[patient_ind]]))
                Cov_Mn = self.Cov_mat(self.est_phi[q,:], self.y_t[patient_ind,:self.T[patient_ind]], grids[i])
                Cov_nM = Cov_Mn.T
                A = np.matmul(Cov_nM, Cov_MM_inv)
                mean = np.matmul(A, self.est_m[patient_ind,:self.T[patient_ind],q].reshape([-1,1]))[0,0]
                if self.nugget:
                    sd = np.sqrt(np.max([self.error, self.est_phi[q,0] + self.est_phi_nugget[q] - np.matmul(A, Cov_Mn)[0,0]]))
                else:
                    sd = np.sqrt(np.max([self.error, self.est_phi[q,0] - np.matmul(A, Cov_Mn)[0,0]]))

                quan[q,:,i] = np.array([mean-1.96*sd, mean, mean+1.96*sd])
        

        if verbose:
            for q in range(self.Q):
                fig=plt.figure()
                # plot latent variables
                plt.scatter(self.y_t[patient_ind,:self.T[patient_ind]], self.est_m[patient_ind,:self.T[patient_ind],q])
                # plot missing variables' time stamp
                plt.axvline(x = T[patient_ind], color='k', linestyle='--')
                # plot 95% quantile for latent process
                plt.plot(np.repeat(grids,2).reshape([-1,2]), quan[q,[0,2],:].T, 'r--')
                # plot mean for latent process
                plt.plot(grids, quan[q,1,:], 'b')
                # plt.show()
                fig.savefig("predicted_latent_process_p{}_{}_{}".format(patient_ind, q, args.method))
        return quan

    def pred_f(self, X, Z, U, theta):
        Cov_MM_inv = np.linalg.inv(self.Cov_mat(theta, Z))
        Cov_Mn = self.Cov_mat(theta, Z, X)
        Cov_nM = Cov_Mn.T
        A = np.matmul(Cov_nM, Cov_MM_inv)
        mean = np.matmul(A, U.reshape([-1, 1]))
        return mean.reshape([-1])

    def softmax(self, x):
        """Compute softmax values for each matrix."""
        if len(x.shape) == 2:
            e_x = np.exp(x - np.max(x, axis=1).reshape(-1,1)*np.ones([1, x.shape[1]]))
            return (e_x / np.sum(e_x, axis = 1).reshape(-1,1)*np.ones([1, x.shape[1]]))
        if len(x.shape) == 1:
            e_x = np.exp(x - np.max(x))
            return e_x/np.sum(e_x)

    def predict_categorical(self, est_x=None):
        if est_x is None:
            # Predict categorical values given latent variables
            res = np.array([np.argmax(self.softmax(np.array([self.pred_f(self.est_x_test, self.est_Z, self.est_mu[:,d,k], self.est_theta[d,:]) for k in range(self.K[d])]).T), axis=1) for d in range(self.D)])
            self.est_y_test = res.T
        else:
            # test
            # print ([self.softmax(np.array([self.pred_f(est_x, self.est_Z, self.est_mu[:,d,k], self.est_theta[d,:]) for k in range(self.K[d])]).T) for d in range(self.D)])
            res = np.array([np.argmax(self.softmax(np.array([self.pred_f(est_x, self.est_Z, self.est_mu[:,d,k], self.est_theta[d,:]) for k in range(self.K[d])]).T), axis=1) for d in range(self.D)])
            self.est_y_test = res.T

def Compute_ind(y, t_num):
    N, T_max, D = y.shape
    y_stacked = y.reshape([-1, D])
    K = np.zeros(D)
    for d in range(D):
        K[d] = len(np.unique(y_stacked[:,d]))
    K_max = max(K)
    y_ind = np.zeros([N, T_max])
    y_dict = {}
    for n in range(N):
        for t in range(t_num[n]):
            for d in range(D):
                y_ind[n,t] += y[n,t,d]*(K_max**d)
            if str(y_ind[n,t]) not in y_dict.keys():
                y_dict[str(y[n,t,:])] = int(y_ind[n,t])
    return y_ind, y_dict

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--method", help="methods: Adam, RSM, Adagrad", type=str, default='Adam')
    parser.add_argument("--dataset", help="name of dataset. two datasets are availble: test_train, test_train_generate, PM_data, stock_monthly", type=str, default='test_train_generate')
    parser.add_argument("--reg", help="regularization", type=float, default=100)
    parser.add_argument("--nugget", help="nugget effects in GPs", action='store_true')
    parser.add_argument("--training_epochs", help="number of training epochs", type=int, default=2000)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.1)
    parser.add_argument("--lower_bound", help="lower_bound of length scale in GP across time", type=float, default=0)
    parser.add_argument("--upper_bound", help="upper_bound of variance of nugget effects", type=float, default=0)
    parser.add_argument("--phi_s2", help="fixed scale paramter in GP of latent process", type=float, default=np.nan)
    parser.add_argument("--phi_l", help="fixed scale length paramter in GP of latent process", type=float, default=np.nan)


    parser.set_defaults(nugget=False)
    args=parser.parse_args()

    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass

    with open('data/'+args.dataset+'.dat', 'rb') as f:
        data_raw = pickle.load(f)
    y_train,  T_train, T_num_train, y_test, T_test = data_raw
    y_train = y_train.astype(np.int32)
    T_train = T_train.astype(np.float32)
    # Scale time
    T_train = T_train/np.max(T_train)
    T_num_train = T_num_train.astype(np.int32)
    # print(y_train.shape, T_train.shape, T_num_train.shape)
    # y_test = y_test.astype(np.int32)
    # T_test = T_test.astype(np.float32)

    # # Create the indicator for training dataset
    # y_train_ind, y_train_dict = Compute_ind(y_train, T_num_train)
    # print(y_train_dict)

    # Generate directories
    os.chdir('./outcome')
    try:
        os.makedirs(args.dataset)
    except:
        pass
    os.chdir('./'+args.dataset)

    try:
        os.makedirs(args.method)
    except:
        pass
    os.chdir('./'+args.method)

    if args.reg!=0:
        try:
            os.makedirs('reg_'+str(args.reg))
        except:
            pass
        os.chdir('./reg_'+str(args.reg))

    try:
        os.makedirs('lb_'+str(args.lower_bound))
    except:
        pass
    os.chdir('./'+'lb_'+str(args.lower_bound))

    if args.nugget:
        try:
            os.makedirs('ub_'+str(args.upper_bound))
        except:
            pass
        os.chdir('./'+'ub_'+str(args.upper_bound))

    if not np.isnan(args.phi_s2) and not np.isnan(args.phi_l):
        try:
            os.makedirs('phi_s2{}phi_l{}'.format(args.phi_s2, args.phi_l))
        except:
            pass
        os.chdir('./'+'phi_s2{}phi_l{}'.format(args.phi_s2, args.phi_l))        

    # if os.path.isfile('CLGP_T_{}.log'.format(args.method)):
    #     os.remove('CLGP_T_{}.log'.format(args.method))
    # Remove all stuffs under current directory
    try:
        os.system("rm -r *")
    except:
        pass

    logging.basicConfig(level=logging.DEBUG, filename='CLGP_T_{}.log'.format(args.method))

    M=20
    Q=2

    # Train data
    np.random.seed(22)
    clgp_t = CLGP_T(M=M, Q=Q, reg=args.reg, nugget=args.nugget)
    clgp_t.fit(y_train, T_train, T_num_train, lamb = 0.001, learning_rate=args.learning_rate, training_epochs = args.training_epochs, method=args.method, verbose=True)
    # save estimated hyper-parameter self.est_theta, self.est_Z, self.est_mu, self.est_L, self.est_m, self.est_s, self.est_phi
    if args.nugget:
        logging.info('theta: {}\n phi: {}\n theta_nugget: {}\n phi_nugget: {}\n'.format(clgp_t.est_theta, clgp_t.est_phi, clgp_t.est_theta_nugget, clgp_t.est_phi_nugget))
    else:
        logging.info('theta: {}\n phi: {}\n'.format(clgp_t.est_theta, clgp_t.est_phi))
    # logging.info('m: {}\n, s: {}\n'.format(clgp_t.est_m, clgp_t.est_s))
    logging.info('elbo: {}'.format(clgp_t.est_elbo))

    # plot all latent variables and inducing points
    est_m_stacked = np.concatenate([clgp_t.est_m[n, :clgp_t.T[n], :] for n in range(clgp_t.N)], axis=0)
    y_train_stacked = np.concatenate([y_train[n, :clgp_t.T[n], :] for n in range(clgp_t.N)], axis=0)
    y_train_stacked_ind = [str(y) for y in y_train_stacked]
    est_m_stacked_df = pd.DataFrame(data = {'x':np.concatenate([est_m_stacked[:,0], clgp_t.est_Z[:,0]]), 'y':np.concatenate([est_m_stacked[:,1], clgp_t.est_Z[:,1]]), 'label':y_train_stacked_ind + ["inducing point" for i in range(M)]})

    # Save resutls
    # self.est_theta, self.est_Z, self.est_mu, self.est_L, self.est_m, self.est_s, self.est_phi
    with open("stock_res.pickle", "wb") as res:
        pickle.dump([clgp_t.est_theta, clgp_t.est_Z, clgp_t.est_mu, clgp_t.est_L, clgp_t.est_m, clgp_t.est_s, clgp_t.est_phi], res)

    fig = sns.lmplot(data=est_m_stacked_df, x='x', y='y', hue='label', fit_reg=False, legend=True, legend_out=True)
    fig.savefig('LS_{}.png'.format(args.method))
    # fig = plt.figure()
    # plt.scatter(est_m_stacked[:,0], est_m_stacked[:,1])
    # plt.scatter(clgp_t.est_Z[:,0], clgp_t.est_Z[:,1])
    # plt.title("All latent variables and inducing points")
    # fig.savefig('LS_{}.png'.format(args.method))

    # predict training data
    print("predict training data")
    clgp_t.predict_categorical(est_m_stacked)
    acc_train = np.mean(clgp_t.est_y_test==y_train_stacked)
    mad_train = np.mean(np.abs(clgp_t.est_y_test - y_train_stacked))
    print("Training data accuracy:{}".format(acc_train))
    print("Training data MAD:{}".format(mad_train))
    logging.info("Training data accuracy: {}".format(acc_train))
    logging.info("Training data MAD: {}".format(mad_train))



    # predict test data
    print("predict testing data")
    clgp_t.predict_latent(T_test)
    clgp_t.predict_categorical()
    # print("predicted result:{}".format(clgp_t.est_y_test.T))
    # print("true result:{}".format(y_test))
    acc_test_cat = np.mean(clgp_t.est_y_test==y_test, axis=0)
    acc_test = np.mean(clgp_t.est_y_test==y_test)
    mad_test = np.mean(np.abs(clgp_t.est_y_test - y_test))
    print("Testing data accuracy for each category:{}".format(acc_test_cat))
    print("Testing data accuracy:{}".format(acc_test))
    print("Testing data MAD:{}".format(mad_test))
    logging.info("Testing data accuracy for each category: {}".format(acc_test_cat))
    logging.info("Testing data accuracy: {}".format(acc_test))
    logging.info("Testing data MAD: {}".format(mad_test))

# 
    # # prdict individual latent process
    # # randomly choose 4 time series to plot the prediction latent process
    # np.random.seed(42)
    # indx_list = np.random.choice(len(T_num_train), 4)
    # for indx in indx_list:
    #     clgp_t.predict_latent_process(indx, T_test, verbose=True)




