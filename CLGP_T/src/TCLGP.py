#!/usr/bin/env python3

# Rui Meng

# 05/17/2019
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


class CLGP_T():
    """
    Variation Inference for temporal categorical latent gaussian process model
    """
    # initialization
    def __init__(self, M=20, Q=2, MC_T=5, error=1e-5, reg=0, nugget=False, method="Adam", training_epochs=2000, learning_rate=0.1, lower_bound=0, upper_bound=0):
        ################################################
        ## M: number of inducing points
        ## Q: latent dimensionality
        ## MC_T: Monte Carlo integration size
        ## error: jitter size
        ## reg: regularization weight
        ## nugget: boolean for nugget effect in GPs
        ################################################
        self.M=M
        self.Q=Q
        self.MC_T=MC_T
        self.error=error
        self.reg=reg
        self.nugget=nugget
        self.method = method
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    # convert vector to lower triangular matrix
    def _L2Ld(self, L):
        idx = np.zeros([self.M, self.M], dtype=np.int32)
        mask = np.zeros([self.M, self.M], dtype=np.bool)
        triu_idx = np.triu_indices(self.M)
        idx[triu_idx] = np.arange((self.M*(self.M+1)/2))
        mask[triu_idx] = True
        Ld = tf.where(mask, tf.gather(L, idx), tf.zeros([self.M, self.M], dtype=L.dtype))
        return tf.transpose(Ld)

    # compute covariance matrix
    def _Cov_mat(self, theta, X1, X2 = None, nugget=0):
        ####################################################
        ## theta: (sigma2, l) hyperparameters of ARD in GP
        ####################################################
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

    # compute covariance matrix
    def _tf_cov(self, x):
        mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        cov_xx += np.diag(np.ones(cov_xx.shape[0]))*self.error
        return cov_xx

    # tranformation function
    def _tranformation_f(self, vec, lower_bound=0, upper_bound=0.5):
        # this function is used to transform variables from R domain to [a,b] domain.float32
        # first, we transform vec to vec0 from R to [0,1] using logistic function
        vec0 = 1.0/(1.0 + tf.exp(-vec))
        return lower_bound+vec0*(upper_bound-lower_bound)

    # create graph
    def _create_graph(self):
        with tf.variable_scope('network_build') as net_build:
            self.lamb = tf.placeholder(tf.float32, name = 'lamb')
            self.ltheta = tf.Variable(tf.zeros([self.D, self.Q + 1], dtype=tf.float32), name="ltheta")
            self.lphi = tf.Variable(tf.zeros([self.Q, 2]), dtype=tf.float32, name = "lphi")
            self.Z = tf.Variable(np.random.randn(self.M, self.Q), dtype=tf.float32, name="Z")
            self.mu = tf.Variable(np.random.randn(self.M, self.D, self.K_max)*0.01, dtype=tf.float32, name="mu")
            self.L = tf.Variable(np.random.randn(self.D, self.M*(self.M+1)//2), dtype=tf.float32, name="L")
            self.m = tf.Variable(tf.zeros([self.N, self.T_max, self.Q], dtype=tf.float32), name="m")
            self.s = tf.Variable(np.log(np.ones([self.N, self.T_max, self.Q])*0.1), dtype=tf.float32, name="s")
            if self.nugget:
                self.ltheta_nugget = tf.Variable(np.log(np.ones(self.D)*0.01), dtype=tf.float32, name="ltheta_nugget")
                self.lphi_nugget = tf.Variable(np.log(np.ones(self.Q)*0.01), dtype=tf.float32, name="lphi_nugget")
                if self.upper_bound !=0:
                    self.theta_nugget = self._tranformation_f(self.ltheta_nugget, upper_bound = self.upper_bound)
                    self.phi_nugget = self._tranformation_f(self.lphi_nugget, upper_bound = self.upper_bound)
                else:
                    self.theta_nugget = tf.exp(self.ltheta_nugget)
                    self.phi_nugget = tf.exp(self.lphi_nugget)
                            
            self.theta = tf.exp(self.ltheta) 

            # Optimize hyper-parameters of GP across time.
            # Convert log scale parameters to standard scale
            # simple transformation.
            # self.phi = tf.exp(self.lphi) 
            # advanced transoformation.
            if self.lower_bound != 0:
                self.phi = tf.stack([tf.exp(self.lphi[:,0]), self._tranformation_f(self.lphi[:,1], lower_bound=self.lower_bound)],axis=1)
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


        with tf.name_scope('Dist_X'):
            # variational distribution of X independent Gaussian with dimension Q by N by T_max.
            QX_loc = [[self.m[n, :self.T[n], q] for n in range(self.N)] for q in range(self.Q)] # Q by N by T_max
            QX_scale_diag = [[tf.exp(self.s)[n, :self.T[n], q] for n in range(self.N)] for q in range(self.Q)]
            QX = tfd.Independent(distribution=tfd.MultivariateNormalDiag(loc = QX_loc, scale_diag=QX_scale_diag), reinterpreted_batch_ndims=0, name = "QX")
            # prior distribution of X from Gauusian process prior with dimentsion Q by N by T
            PX_loc = [[tf.zeros(self.T[n]) for n in range(self.N)] for q in range(self.Q)]
            PX_covariance_matrix = [[cov_mat_x_list[q*self.N+n] for n in range(self.N)] for q in range(self.Q)]
            PX = tfd.Independent(distribution=tfd.MultivariateNormalFullCovariance(loc = PX_loc, covariance_matrix= PX_covariance_matrix), reinterpreted_batch_ndims=0, name = "PX")

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
            self.KL_X = tf.reduce_sum(tf.distributions.kl_divergence(QX, PX, name = 'KL_X'))
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
                # print ("QX has been sampled.")

                ### Sample U
                sampled_U = []
                for d in range(self.D):
                    sampled_eps_U_d = tf.random_normal([self.M, self.K[d]])
                    sampled_U_d = self.mu[:, d, :self.K[d]] + tf.matmul(Ld_list[d], sampled_eps_U_d)
                    paddings = np.array([[0, 0], [0, self.K_max-self.K[d]]])
                    sampled_U.append(tf.pad(sampled_U_d, paddings))
                # print ("QU has been sampled.")
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
                # print ("QF has been sampled.")
                # print ("{}th Monte Carlo iteration has been completed.".format(tt))
            self.Comp_F = self.Comp_F/self.MC_T
            tf.summary.scalar("summary/Comp_F", self.Comp_F)

    # create optimizer
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

    # create summary
    def _create_summary(self):
        self.summ =  tf.summary.merge_all()

    # fitting function
    def fit(self, Y_train, T_train, T_num_train, display_step=5, learning_rate=0.1, lamb=1, lamb_inc=0.001, training_epochs=2000, model_path='/tmp/model_tem.ckpt', hist_path='./logs/tem', method='Adam', verbose=False):
        ######################################
        ## Y_train: N x T_max x D
        ## T_train: N x T_max
        ## T_num_train: N
        ## lamb: initial weight for Stimulating Leaning. If lamb = 1, we do not use simulating learning.
        ######################################
        
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
            if epoch % display_step == 0:
                print("{} epochs have been completed.".format(epoch))
                # print training information
                lamb0, elbo0, theta0, phi0, Z0, KL_U0, KL_X0, KL_ZX0, Comp_F0, summary = self.sess.run([self.lamb, self.elbo, self.theta, self.phi, self.Z, self.KL_U, self.KL_X, self.KL_ZX, self.Comp_F, self.summ], feed_dict={self.lamb: lamb})
            #     print("Epoch: {}".format(epoch+1))
                print("elbo: {}, F: {}, KL_U:{}, KL_X:{}, KL_ZX:{}, phi:{}".format(elbo0, Comp_F0, KL_U0, KL_X0, KL_ZX0, phi0))
            # writer.add_summary(summary, epoch)
            lamb = min(1, lamb_inc+lamb)
            # elbo_hist.append(elbo0)

        if verbose:
            fig=plt.figure()
            plt.plot(elbo_hist)
            plt.title('Elbo_trace')
            fig.savefig('EBLO_trace_{}.png'.format(self.method))
            plt.close(fig)

        # Close history, saver, model saver and session
        writer.close()
        print("History has been saved under {}".format(hist_path))
        # Save model weights to disk
        saver.save(self.sess, model_path)
        print("Model has beend saved under{}".format(model_path))

        # Save all model parameters
        if self.nugget:
            self.est_theta, self.est_Z, self.est_mu, self.est_L, self.est_m, self.est_s, self.est_phi, self.est_theta_nugget, self.est_phi_nugget = self.sess.run([self.theta, self.Z, self.mu, self.L, self.m, self.s, self.phi, self.theta_nugget, self.phi_nugget])
        else:
            self.est_theta, self.est_Z, self.est_mu, self.est_L, self.est_m, self.est_s, self.est_phi = self.sess.run([self.theta, self.Z, self.mu, self.L, self.m, self.s, self.phi])
        self.sess.close()

    # compute covariance matrix
    def Cov_mat(self, theta, X1, X2 = None, nugget=0):
        #######################################################
        # theta (sigma2, l): hyperparameters of ARD in GPs 
        #######################################################
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
            if self.nugget:
                cov_mat += np.diag(np.ones(X1.shape[0])*nugget)
        return cov_mat

    # predict latent variables given new time stamps
    def predict_latent(self, T_test):
        ####################################################################
        ## Predict latent variables on T_test using posterior mean
        ## T_test: N
        ####################################################################
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

    # predict latent process for individual index
    def predict_latent_process(self, patient_ind, n_grids=100, verbose=False):
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
                # plot 95% quantile for latent process
                plt.plot(np.repeat(grids,2).reshape([-1,2]), quan[q,[0,2],:].T, 'r--')
                # plot mean for latent process
                plt.plot(grids, quan[q,1,:], 'b')
                # plt.show()
                fig.savefig("predicted_latent_process_p{}_{}_{}".format(patient_ind, q, self.method))
        return quan

    # compute predictive posterior mean
    def pred_f(self, X, Z, U, theta):
        Cov_MM_inv = np.linalg.inv(self.Cov_mat(theta, Z))
        Cov_Mn = self.Cov_mat(theta, Z, X)
        Cov_nM = Cov_Mn.T
        A = np.matmul(Cov_nM, Cov_MM_inv)
        mean = np.matmul(A, U.reshape([-1, 1]))
        return mean.reshape([-1])

    # softmax function
    def softmax(self, x):
        """Compute softmax values for each matrix."""
        if len(x.shape) == 2:
            e_x = np.exp(x - np.max(x, axis=1).reshape(-1,1)*np.ones([1, x.shape[1]]))
            return (e_x / np.sum(e_x, axis = 1).reshape(-1,1)*np.ones([1, x.shape[1]]))
        if len(x.shape) == 1:
            e_x = np.exp(x - np.max(x))
            return e_x/np.sum(e_x)

    # predict categorical values
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



# compute index
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
    return y_ind, y_dic

if __name__ == "__main__":
    pass