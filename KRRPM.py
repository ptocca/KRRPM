from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
import scipy.linalg
import scipy.sparse

import logging

import resource

######
# Needed to convert the output of DPOTRI into a symmetric matrix
# Used by spd_inverse_cholesky_inplace
######
import numba

@numba.jit('float64[:,:](float64[:,:])',nopython=True,nogil=True)
def make_sym_from_upper_inplace(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            A[j,i]=A[i,j]
    return A
######

#######
# Inverse of a semi positive definite matrix with the Cholesky decomposition
#
# This uses almost no extra memory.
# The argument is overwrittent with the result
#######
def spd_inverse_cholesky_inplace(a):
    """Inverse of a semi positive definite symmetric matrix via Cholesky factorisation
    NOTE: this overwrites the argument!!!
    """
    a,res1 = scipy.linalg.lapack.dpotrf(a.T,0,0,overwrite_a=1)  # The .T makes it FORTRAN-contiguous
    a,res2 = scipy.linalg.lapack.dpotri(a,0,overwrite_c=1)
    logging.info("DPOTRF info, DPOTRI info: %d,%d"%(res1,res2))
    return make_sym_from_upper_inplace(a)


######
# Low-mem matrix multplication
#
# numpy.matmul() has an output parameter, but allocates a temporary of the same size as the output matrix.
# This is evident only when you look at maxRSS.
# This hidden behaviour must have to do with how BLAS DGEMM() works.
#
# matmul_lowmem() performs the multiplication breaking it down across the columns of the second matrix.
# It requires a fraction of the memory that numpy.matmul() requires behind the scenes. However, it takes more CPU time.
#
# The b matrix is overwritten with the result
#
# Both a and b must be FORTRAN-ordered
######
def matmul_lowmem(a,b,bunch_size = 200):
    tmp = np.empty(shape=(b.shape[0],bunch_size),order='F')
    for i in range(0,b.shape[1],bunch_size):
        bunch = min(bunch_size,b.shape[1]-i)
        np.copyto(tmp[:,:bunch],b[:,i:i+bunch])
        scipy.linalg.blas.dgemm(1.0,a,tmp[:,:bunch],0.0,b[:,i:i+bunch],overwrite_c=1)
################

from sklearn.metrics.pairwise import check_pairwise_arrays,euclidean_distances,linear_kernel

def euclidean_distances_lowmem(X, Y, squared=True, bunch_size=10000):
    m = X.shape[0]
    p = X.shape[1]
    n = Y.shape[0]
    D = np.empty(shape=(m,n))
    for i in range(0, m, bunch_size):
        bunch_i = min(bunch_size, m - i)
        for j in range(0, n, bunch_size):
            bunch_j = min(bunch_size, n - j)
            D[i:i+bunch_i, j:j+bunch_j] = euclidean_distances(X[i:i+bunch_i], Y[j:j+bunch_j], squared=squared)
    return D

# Taken from sklearn.metrics.pairwise, this avoids a temporary
def rbf_kernel(X, Y=None, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <rbf_kernel>`.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float, default None
        If None, defaults to 1.0 / n_features
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances_lowmem(X, Y, squared=True)
    np.multiply(K,-gamma,out=K)      # Avoid the temporary
    np.exp(K, out=K)  # exponentiate K in-place
    return K


KERNELS_FUNCS = {
    "rbf": rbf_kernel,
    "linear": linear_kernel
}

def compute_kernels(X,Y=None,metric=None,**kernel_kwargs):
    try:
        metric = KERNELS_FUNCS[metric]
    except:
        pass

    if callable(metric):
        K = metric(X,Y,**kernel_kwargs)     # More efficient semantics than in pairwise kernels
    else:
        K = pairwise_kernels(X, Y, metric=metric,
                             **kernel_kwargs)
    return K

class KRRPM(BaseEstimator, RegressorMixin):
    """Computes the Predictive Distributions according the Kernel Ridge Regression Predictive Machine"""
    def __init__(self, a, kernel, lb=None, ub=None, max_pd_pts=1000, kernel_kwargs={}):
        self.kernel = kernel
        self.a = a
        self.lb = lb
        self.ub = ub
        self.max_pd_pts = max_pd_pts
        self.kernel_kwargs = kernel_kwargs
        self.ss = None
        self.y_offset = None
        self.center_y = None
        self.H_diag = None
        self.K_inv = None

    def fit(self, X, y, center_y=True):
        self.y = np.array(y).reshape(-1, 1)

        self.center_y = center_y

        if self.center_y:
            self.y_offset = np.average(self.y)
            self.y -= self.y_offset
            logging.info("Centered y by: %f" % self.y_offset)

        if scipy.sparse.issparse(X):
            self.ss = MaxAbsScaler()  # This preserves sparsity
        else:
            self.ss = StandardScaler()
        self.X = self.ss.fit_transform(X)

        logging.info("Gram matrix calculation")

        K = compute_kernels(self.X, metric=self.kernel,
                             **self.kernel_kwargs).T  # the kernel matrix (step 1 of preprocessing in the paper)
        logging.info("average Kernel: %f", np.average(K))

        logging.info("Hat matrix calculation")
        # K_reg =  K + self.a * np.eye(self.X.shape[0], dtype=np.float64)
        # This computes the same as some but avoids large temporary matrices
        K_reg = K.copy()
        for i in range(K_reg.shape[0]):
            K_reg[i,i] += self.a

        self.K_inv = spd_inverse_cholesky_inplace(K_reg)                  # step 2 of preprocessing in the paper

        logging.info("After inversion: MaxRSS %d"%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # H = np.matmul(K, self.K_inv, out=K)  # the hat matrix (step 3 of preprocessing in the paper)
        matmul_lowmem(self.K_inv, K, bunch_size=100)   # this replaces the commented statement above


        logging.info("After matmul_lowmem(): MaxRSS %d"%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        H = K                                              # use the H name just for consistency with the paper

        self.H_diag = np.diag(H).copy()    # extract the diagonal as a vertical vector
        # We need to copy() because np.diag() returns a view, so the memory for H would remain allocated
        self.y_hat = H @ self.y  # y hat  (step 4 of preprocessing in the paper)
        logging.info("Fitting completed")
        del H
        logging.info("End of fit() MaxRSS %d"%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


    def _predict_point(self, k, kappa):
        epsilon = np.matmul(self.K_inv,k)  # prediction step 2 in the paper

        semi_d = np.dot(k.T,epsilon)       # prediction step 3 in the paper  scalar

        d = 1 / (kappa + self.a - semi_d)  # prediction step 4 in the paper  scalar

        one_minus_h = 1 - self.H_diag + self.a * d * np.square(epsilon)  # prediction step 5 in the paper

        hat_test = np.dot(epsilon.T,self.y)  # prediction step 6 in the paper
        sqrt_one_minus_h = np.sqrt(one_minus_h)

        A = np.sqrt(self.a * d) * hat_test + (self.y.ravel() - self.y_hat.ravel() + self.a * d * hat_test * epsilon) / sqrt_one_minus_h
        B = np.sqrt(self.a * d) + (self.a * d * epsilon) / sqrt_one_minus_h

        C = np.divide(A,B)
        if self.center_y:
            C += self.y_offset

        distribution = np.sort(C)
        if self.lb is not None or self.ub is not None:
            distribution = distribution.clip(self.lb, self.ub)
        middle_index = int(C.shape[0]/2)

        C_srt = distribution[middle_index]

        if (not (self.max_pd_pts is None)) and (len(distribution) > self.max_pd_pts):
            idxs = np.linspace(0, len(distribution)-1, self.max_pd_pts).astype(int)
            distribution = distribution[idxs]

        return C_srt,distribution


    def predict(self,X):
        X_scaled = self.ss.transform(X)

        ks = compute_kernels(self.X,
                             X_scaled,
                             metric=self.kernel,
                             **self.kernel_kwargs)  # the kernel vector (prediction step 1 in the paper)

        kappas = np.zeros(X_scaled.shape[0])
        for i in range(X_scaled.shape[0]):
            if scipy.sparse.issparse(X_scaled):
                x = X_scaled[i]
            else:
                x = np.atleast_2d(X_scaled[i])
            kappas[i] = compute_kernels(x,
                         metric=self.kernel,
                         **self.kernel_kwargs)  # the kernel scalar (prediction step 1 in the paper)

        logging.info("Test object kernels done.")

        y_hat = np.zeros(shape=(X.shape[0],1))
        self.predicted_distributions = []
        for i,(k,kappa) in enumerate(zip(ks.T,kappas)):
            if (i%1000) == 0 and i!=0:
                logging.info("Predicted %d objects.",i)
            y_hat[i],d = self._predict_point(k,kappa)
            self.predicted_distributions.append(d)
        self.predicted_distributions = np.array(self.predicted_distributions)
        logging.info("Predicted %d objects. Done.",i)
        return y_hat
