'''
Created on 18Nov.,2016

@author: sforaz
'''
from scipy import signal
import matplotlib.pyplot as plot
import glob
import os
import scipy.io
from scipy.signal import argrelextrema, resample
import numpy as np
from sklearn import mixture
import sys
from sklearn.covariance import graph_lasso  # , GraphLassoCV
from scipy.spatial.distance import pdist, squareform
# from numbapro import jit, float32


def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor


class DynamicFC:

    def __init__(self, directory, good_components, ica=True):

        self.directory = directory
        self.good_components = good_components
        self.ica = ica
        self.resampled_mat = []

    @staticmethod
    def mat2vec(mat):
        """
        vec = mat2vec(mat)
        returns the lower triangle of mat
        mat should be square
        """
        [n, m] = mat.shape

        if n != m:
            print('Error: mat must be square!')
            return False

        temp = np.ones((n, n))
        #  find the indices of the lower triangle of the matrix
        IND = np.where((temp-np.triu(temp)) > 0)

        vec = mat[IND]

        return vec

    @staticmethod
    def vec2mat(vec, symmetric):
        """
        mat = vec2mat(vec,1)
        returns full matrix starting from a vector
        """
        N = len(vec)
        n_comp = int(0.5+np.sqrt(1+8*N)/2)
        mat = np.zeros((n_comp, n_comp))

        temp = np.ones((n_comp, n_comp))

        IND = np.where((temp-np.triu(temp)) > 0)

        mat[IND] = vec

        if symmetric:
            tempmat = np.flipud(np.rot90(mat))
            tempmat[IND] = vec
            mat = np.flipud(np.rot90(tempmat))

        return mat

    def load_subjects(self):

        good_comp = [x-1 for x in self.good_components]

        if self.ica:
            list_sub = glob.glob(self.directory + '/_ica_c*-1.mat')
            list_sub = sorted(list_sub)

            mat_1 = scipy.io.loadmat(list_sub[0])
            mat_1 = mat_1['tc']
        else:
            list_sub = glob.glob(self.directory+'/*.txt')
            list_sub = sorted(list_sub)

            mat_1 = np.loadtxt(list_sub[0])

        n_subjects = len(list_sub)
        n_timepoints = mat_1.shape[0]
        n_ic_components = len(good_comp)

        all_sub = np.zeros((n_timepoints*n_subjects, n_ic_components))

        for i, sub in enumerate(list_sub):
            if self.ica:
                mat = scipy.io.loadmat(sub)
                mat = mat['tc']
            else:
                mat = np.loadtxt(sub)

            mat = mat[:, good_comp]
            all_sub[i*n_timepoints:(i+1)*n_timepoints, :] = mat

        self.all_subs = all_sub
        self.tp = n_timepoints
        self.list_subs = list_sub

    def windowed_fc(self, window_tp=60, step=1, sigma=20, method='corr'):

        subs = self.all_subs
        overlap = window_tp-step
        n_ic_components = int(np.size(subs, 1))
        n_sub = int(np.size(subs, 0)/self.tp)
        n_window = int((self.tp-overlap)/(window_tp-overlap))
        cov_mat = (np.zeros((n_sub, n_window-1, (n_ic_components *
                                                 (n_ic_components-1))//2)))
        # netmat = np.zeros((n_sub, n_ic_components*n_ic_components))
        gaus_win = signal.gaussian((window_tp), std=sigma)
        rect = np.ones(window_tp)
        conv_gaus = signal.convolve(rect, gaus_win, 'same')
        conv_gaus = conv_gaus/conv_gaus.max()

        for j in range(n_sub):
            sub = subs[j*self.tp:(j+1)*self.tp, :]
            for i in range(n_window-1):
                sub_window = sub[i*(window_tp-overlap):(i+1) *
                                 (window_tp-overlap)+overlap, :]
                sub_window = np.multiply(
                    sub_window, np.tile(conv_gaus, (np.size(sub_window, 1),
                                                    1)).T)
                if method == 'corr':
                    cov_1 = np.corrcoef(sub_window.T)
                    cov_1[np.eye(n_ic_components) > 0] = 0
                    cov_1 = self.mat2vec(cov_1)
                    cov_1 = np.arctanh(cov_1)
                    # cov_1=cov_1-np.mean(cov_1)
                    # cov_1=cov_1/np.std(cov_1)
                    cov_mat[j, i, :] = cov_1
                if method == 'part_corr':
                    cov_1 = np.cov(sub_window.T)
                    cov_1 = - np.linalg.inv(cov_1)
                    cov_1 = (
                        cov_1 / np.tile(np.sqrt(np.abs(np.diag(cov_1))),
                                        (n_ic_components, 1)).T)/np.tile(
                                            np.sqrt(np.abs(np.diag(cov_1))),
                                            (n_ic_components, 1))
                    cov_1[np.eye(n_ic_components) > 0] = 0
                    cov_1 = self.mat2vec(cov_1)
                    cov_1 = np.arctanh(cov_1)
                    # cov_1=cov_1-np.mean(cov_1)
                    # cov_1=cov_1/np.std(cov_1)
                    cov_mat[j, i, :] = cov_1
                elif method == 'reg_pc':
                    cov_true = np.cov(sub_window.T)
                    cov_true = cov_true/np.mean(np.diag(cov_true))
                    # GL = GraphLassoCV()
                    # gl_fit = GL.fit(cov_true)
                    # cov_1 = gl_fit.get_precision()
                    cov, cov_1 = graph_lasso(cov_true, 0.00001, max_iter=500,
                                             tol=0.005)
                    cov_1 = -(
                        (cov_1 / np.tile(np.sqrt(np.abs(np.diag(cov_1))),
                                         (n_ic_components, 1)).T) /
                        np.tile(np.sqrt(np.abs(np.diag(cov_1))),
                                (n_ic_components, 1)))
                    cov_1[np.eye(n_ic_components) > 0] = 0
                    cov_1 = self.mat2vec(cov_1)
                    cov_1 = np.arctanh(cov_1)
                    cov_mat[j, i, :] = cov_1
                elif method == 'ridge_pc':
                    cov_true = np.cov(sub_window.T)
                    cov_true = cov_true/np.sqrt(np.mean(np.diag(
                        np.square(cov_true))))
                    cov_1 = - np.linalg.inv(cov_true+0.1*np.eye(
                        n_ic_components))
                    cov_1 = (
                        cov_1 / np.tile(np.sqrt(np.abs(np.diag(cov_1))),
                                        (n_ic_components, 1)).T)/np.tile(
                                           np.sqrt(np.abs(np.diag(cov_1))),
                                           (n_ic_components, 1))
                    cov_1[np.eye(n_ic_components) > 0] = 0
                    cov_1 = self.mat2vec(cov_1)
                    cov_1 = np.arctanh(cov_1)
                    # cov_1=cov_1-np.mean(cov_1)
                    # cov_1=cov_1/np.std(cov_1)
                    cov_mat[j, i, :] = cov_1

        self.covariance_mat = cov_mat

    def subsampling(self):

        cov_mat = self.covariance_mat
        # self.resampled_mat = []

        for i in range(cov_mat.shape[0]):

            mat = cov_mat[i, :, :]
            if mat.shape[0] > 100:
                mat = resample(mat, 100, axis=0)
                var = np.var(mat, axis=1)
                local_max = argrelextrema(var, np.greater)
                local_max = np.squeeze(local_max)
                for m in local_max:
                    self.resampled_mat.append(mat[m, :])
            else:
                for m in range(mat.shape[0]):
                    self.resampled_mat.append(mat[m, :])

        self.resampled_mat = np.asarray(self.resampled_mat)

        # self.resampled_mat = resampled_mat

    def find_num_clusters_gmm(self, components=13):
            """Function to find the best parameters for your GMM.
           Input:
               components: number of components with which you want to test
               your model. For example, if you choose 10,
               the function will fit up to 10 components into your data and
               returns the best choice of parameters.
           Usage:

           bic,best_parameters,best_gmm=dynamic.find_num_clusters_gmm(10)
            """
            if self.resampled_mat.any():
                mat = self.resampled_mat
            else:
                mat = self.covariance_mat

            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, int(components))
            cv_types = ['spherical', 'tied', 'diag', 'full']
            # cv_types = ['full']

            for cv_type in cv_types:
                for n_components in n_components_range:
                    # Fit a mixture of Gaussians with EM
                    sys.stdout.write(
                        "\r" + cv_type+', components: '+str(n_components))
                    # sys.stdout.flush()
                    gmm = mixture.GaussianMixture(n_components=n_components,
                                                  covariance_type=cv_type)
                    gmm.fit(mat)
                    bic.append(gmm.bic(mat))
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm
                        best_parameters = [n_components, cv_type]
                    sys.stdout.flush()
                # sys.stdout.flush()

            bic = np.array(bic)
            best_parameters = np.array(best_parameters)

            self.bic = np.array(bic)
            self.best_parameters = np.array(best_parameters)
            self.best_gmm = best_gmm
            self.best_gmm_weights = best_gmm.weights_
            self.best_gmm_mean = best_gmm.means_

    def gmm_clustering(self):
        """Function that apply GMM clustering to the whole dataset.
           Inputs:
               - n_components: the number of cluster you want to fit into
                your dataset.
               - method: clustering method (between 'spherical', 'diag', 'tied'
                and 'full').
               Usually both these inputs should be found using
               find_num_clusters_gmm.
           Usage:

           dynamic.GMM_clustering(9,'diag')
        """
        self.num_clusters = int(self.best_parameters[0])
        n_components = int(self.best_parameters[0])
        method = self.best_parameters[1]
        self.cov_mat_reshaped = (
            np.reshape(self.covariance_mat,
                       (self.covariance_mat.shape[0] *
                        self.covariance_mat.shape[1],
                        self.covariance_mat.shape[2]), order='F'))
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=method)
        gmm.fit(self.resampled_mat)
        self.gmm_weights = gmm.weights_
        self.gmm_means = gmm.means_
        gmm_full = mixture.GaussianMixture(
            n_components=n_components, covariance_type=method)
        gmm_full.means_ = gmm.means_
        gmm_full.fit(self.cov_mat_reshaped)
        self.gmm_full_weights = gmm_full.weights_
        self.gmm_full_means = gmm_full.means_
        self.predicted_labels = gmm_full.predict(
            self.cov_mat_reshaped)
        self.FC_states = np.reshape(
            self.predicted_labels, (self.covariance_mat.shape[0],
                                    self.covariance_mat.shape[1]), order='F')

    def static_fc(self):
        """Function to compute the static FC.
           Usage:

           statFC=dynamic.static_fc()
        """

        n_timepoints = self.tp
        n_sub = self.all_subs.shape[0]//n_timepoints
        corr_mat = np.zeros((n_sub, self.all_subs.shape[1],
                             self.all_subs.shape[1]))
        for j in range(n_sub):
            sub = self.all_subs[j*n_timepoints:(j+1)*n_timepoints, :]
            cov_1 = np.corrcoef(sub.T)
            cov_1[np.eye(self.all_subs.shape[1]) > 0] = 0
            cov_1 = np.arctanh(cov_1)
            corr_mat[j, :, :] = cov_1
        stat_fc = np.mean(corr_mat, axis=0)
        self.static_FC = stat_fc

    def transition_mat(self, a):

        Nwin = len(a)
        k = self.num_clusters
        TM = np.zeros((k, k))
        for t in range(1, Nwin):
            TM[a[t-1], a[t]] = TM[a[t-1], a[t]]+1

        for jj in range(k):
            if np.sum(TM[jj, :] > 0):
                TM[jj, :] = TM[jj, :]/np.sum(a[1: Nwin-1] == jj)
            else:
                TM[jj, jj] = 1

        return TM

    def mean_dwell_time(self, a):

        Nwin = len(a)
        k = self.num_clusters
        MDT = np.zeros((1, k))

        for jj in range(k):
            start_t = np.where(np.diff((a == jj).astype(int)) == 1)[0]
            end_t = np.where(np.diff((a == jj).astype(int)) == -1)[0]
            if a[0] == jj:
                start_t = np.concatenate((np.array([0]), start_t))
            if a[-1] == jj:
                end_t = np.concatenate((end_t, np.array([Nwin])))
            MDT[0, jj] = np.mean(end_t-start_t)
            if (not end_t.size) and (not start_t.size):
                MDT[0, jj] = 0

        return MDT

    def gen_cluster_stats(self):

        M = self.all_subs.shape[0]/self.tp
        TM_tot = np.zeros((M, self.num_clusters, self.num_clusters))
        MDT_tot = np.zeros((M, self.num_clusters))
        for ii in range(M):
            TM = self.transition_mat(self.FC_states.T[:, ii])
            MDT = self.mean_dwell_time(self.FC_states.T[:, ii])
            TM_tot[ii, :, :] = TM
            MDT_tot[ii, :] = MDT

        return TM_tot, MDT_tot

    def save_results(self, save_dir, output='all', gofigure=0):
        """Function to save your Dynamic Connectivity results.
           Inputs:
               - save_dir: path where you want to save the results. It creates
                the directory you input (if it's not present).
               - output: '1' to save just FC states switching plots
                           (number of plots=number of subjects).
                         '2' to save just the n_comp X n_comp cluster centroids
                             matrices (#matrices=#clusters).
                         '3' to save just the Static FC matrix.
                         '4' to save just the text files with the clusters
                           mean, cluster weights and switching states for each
                             subject.
                         'all' to save all the previus results.
               - gofigure: input gofigure=1 if you want to see the figures.
                   default no.
           Usage:

           dynamic.save_results('path/to/save_dir','all')
        """
        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)

        if output == '1' or output == 'all':

            try:
                for i in range(self.FC_states.shape[0]):
                    plot.plot(self.FC_states[i, :])
                    basename = (
                        'FC_states_switching_sub_{}'
                        .format(str(i).zfill(3)))
                    plot.xlim(0, self.FC_states.shape[1])
                    plot.ylim(-1, len(self.gmm_full_weights)+1)
                    plot.savefig(save_dir+basename, bbox_inches='tight')
                    plot.close()
            except:
                print('You have not generated FC states switching yet!')

        if output == '2' or output == 'all':

            try:
                for i in range(self.gmm_full_means.shape[0]):
                    basename = 'Centroids_Cluster_'+str(i+1)
                    mat = self.vec2mat(self.gmm_full_means[i, :], 1)
                    plot.pcolor(mat, vmin=-0.7, vmax=0.7)
                    plot.xlim(0, mat.shape[0])
                    plot.ylim(0, mat.shape[0])
                    plot.colorbar()
                    w = '{0:.4f}'.format(self.gmm_full_weights[i])
                    plot.title(basename.replace('_', ' ')+' (Weight: '+w+')')
                    plot.savefig(save_dir+basename, bbox_inches='tight')
                    if gofigure:
                        plot.show()
                    else:
                        plot.close()
            except:
                print('You have not generated mean cluster centroids yet!')

        if output == '3' or output == 'all':

            try:
                plot.pcolor(self.static_FC, vmin=-0.5, vmax=0.5)
                plot.xlim(0, self.static_FC.shape[0])
                plot.ylim(0, self.static_FC.shape[0])
                plot.colorbar()
                plot.savefig(save_dir+'/Static_FC.png', bbox_inches='tight')
                if gofigure:
                    plot.show()
                else:
                    plot.close()
            except:
                print('You have not generated Static FC yet!')

        if output == '4' or output == 'all':

            np.savetxt(save_dir+'FC_states.txt', self.FC_states, fmt='%i')
            np.savetxt(save_dir+'Full_GMM_Weights.txt',
                       self.gmm_full_weights, fmt='%f')
            np.savetxt(save_dir+'Full_GMM_Means.txt',
                       self.gmm_full_means, fmt='%f')
