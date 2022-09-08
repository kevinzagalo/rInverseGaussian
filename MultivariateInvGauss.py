from rInvGauss import rInvGauss
from math import sqrt
from numpy import std, array

class MultivariateInvGauss:

    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov
        self.modes = None
        self.smooth = None

    def fit(self, X):
        XX = array(X)
        n, d = XX.shape
        for j in range(d):
            rIG = rInvGauss().fit(XX[:, j])
            res_dict = rIG.get_parameters()
            self.modes[j] = res_dict['mode']
            self.smooth[j] = res_dict['smooth']

        tmp = cumsum()
        for j in range(d):
            self.rho[j] =  std(XX[:, j], ddof=1) /
