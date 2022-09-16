import numpy as np
from math import sqrt, log, exp
from scipy.optimize import minimize, fmin_bfgs, minimize_scalar, root_scalar
from rInverseGaussian.rInvGaussMixture import rInvGaussMixture
from sklearn.cluster import KMeans


class RTInvGaussMixture(rInvGaussMixture):

    def __init__(self, n_components, cv_init, max_iter=100, tol=1e-4, modes_init=None, backlog_init=None,
                 weights_init=None, verbose=False, utilization=None):

        super().__init__(n_components=n_components, tol=tol, max_iter=max_iter, modes_init=modes_init,
                         cv_init=cv_init, weights_init=weights_init, verbose=verbose)
        self.utilization = utilization

        if cv_init is not None and (isinstance(cv_init, int) or isinstance(cv_init, float)):
            self.cv_ = [cv_init] * n_components

        self.backlog_ = backlog_init
        if cv_init and modes_init:
            self.backlog_ = [self._backlog(m, cv_init) for m in modes_init]

    def _mode(self, backlog=None):
        return sqrt((backlog / (1 - self.utilization)) ** 2 + (1.5 * self.cv_[0]) ** 2) - 1.5 * self.cv_[0]

    def _backlog(self, theta, gamma):
        return (1 - self.utilization) * sqrt(theta) * sqrt(theta + 3 * gamma)
    
    def dtheta_dbacklog(self, backlog):
        return backlog / (1 - self.utilization) ** 2 / sqrt((backlog / (1 - self.utilization)) ** 2 + (1.5 * self.cv_[0]) ** 2)

    def d2theta_dbacklog2(self, backlog):
        return (1.5 * self.cv_[0]) ** 2 * (1 - self.utilization) / (
                backlog ** 2 + (1.5 * self.cv_[0] * (1 - self.utilization)) ** 2) ** 1.5

    def _update_params(self, XX, zz, x0, method='dogleg'):
        hess_LL = lambda y: -self.d2theta_dbacklog2(y) * \
                            self._derivative_complete_likelihood(XX, zz, self._mode(y), self.cv_[0])[
                                0] - self.dtheta_dbacklog(y) ** 2 * \
                            self._second_derivative_complete_likelihood(XX, zz, self._mode(y), self.cv_[0])[0, 0]
        grad_LL = lambda y: -self.dtheta_dbacklog(y) * \
                            self._derivative_complete_likelihood(XX, zz, self._mode(y), self.cv_[0])[0]
        LL = lambda y: -self._complete_likelihood(XX, zz, self._mode(y), self.cv_[0])
        if method == 'dogleg':
            res = minimize(fun=LL, method=method, x0=x0, jac=grad_LL, hess=hess_LL)['x']
        elif method == 'BFGS':
            res = minimize(fun=LL, method=method, x0=x0, jac=grad_LL)['x']
        elif method == 'minimize_scalar':
            res = minimize_scalar(LL).x
        elif method == 'newton':
            res = root_scalar(grad_LL, x0=x0, fprime=hess_LL, method='newton').root
        else:
            raise("method unknown. Try 'dogleg', 'BFGS', 'minimize_scalar' or 'newton'.")
        if isinstance(res, list):
            return res[0]
        else:
            return res

    def initialize(self, X, method='kmeans'):
        if method == 'kmeans':
            kmeans = KMeans(self._n_components).fit(X.reshape(-1, 1))

        z = np.zeros((len(X), self._n_components))
        if self.weights_ is None:
            for i, j in enumerate(kmeans.predict(X.reshape(-1, 1))):
                z[i, j] = 1
            self.weights_ = np.mean(z, axis=0).tolist()

        self.modes_ = kmeans.cluster_centers_.reshape(-1)
        self.backlog_ = [self._backlog(m, self.cv_[0]) for m in self.modes_]

        if self.cv_ is None:
            self.cv_ = [1.]

    def _M_step(self, X, z, method='dogleg'):
        # M-step
        self.weights_ = np.mean(z, axis=0).tolist()
        self.backlog_ = np.array(self.backlog_).T
        for j in range(self._n_components):
            self.backlog_[j] = self._update_params(X, z[:, j], self.backlog_[j], method=method)
            self.modes_[j] = self._mode(self.backlog_[j])
        return 0

    def aic(self, X):
        return 2 * len(X) * self.score(X) - (2 * self._n_components - 1) * 2

    def bic(self, X):
        return 2 * self.score(X) - (2 * self._n_components - 1) * np.log(len(X))

    def get_parameters(self):
        return {'weights': tuple(self.weights_), 'modes': tuple(self.modes_), 'backlog': tuple(self.backlog_),
                'utilization': self.utilization, 'cv': self.cv_[0], 'n_components': self._n_components}
