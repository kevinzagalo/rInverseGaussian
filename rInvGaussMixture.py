import numpy as np
from math import log
from scipy.optimize import minimize, fmin_bfgs
from tqdm import trange
from rInverseGaussian.rInvGauss import rInvGauss
from sklearn.cluster import KMeans
from scipy.stats import kstest, chi2


class rInvGaussMixture:

    def __init__(self, n_components, max_iter=100, tol=1e-4, modes_init=None,
                 cv_init=None, weights_init=None, verbose=False):
        self.tol = tol
        self.n_iter_ = max_iter
        self.verbose = verbose
        self._n_components = n_components

        if weights_init is not None:
            assert len(weights_init) == self._n_components, 'Weights lengths should be equal to n_components'

        if modes_init is not None:
            assert len(modes_init) == self._n_components, 'Modes lengths should be equal to n_components'

        self.modes_ = modes_init
        self.cv_ = cv_init
        self.weights_ = weights_init
        self.converged_ = False
        self.fitted_ = False

    def _mean(self, k):
        return np.sqrt(self.modes_[k] * (3 * self.cv_[k] + self.modes_[k]))

    def _proba_components(self, x):
        return [pi_j * rInvGauss(self.modes_[j], self.cv_[j]).pdf(x) for j, pi_j in enumerate(self.weights_)]

    def pdf(self, x):
        return sum(self._proba_components(x))

    def cdf(self, x):
        return [pi_j * rInvGauss(self.modes_[j], self.cv_[j]).cdf(x) for j, pi_j in enumerate(self.weights_)]

    def quantile(self, alpha, component):
        return rInvGauss(self.modes_[component], self.cv_[component]).quantile(alpha)

    def normalize(self, x, y):
        return (x - self._mean(y)) ** 2 / (self.cv_[0] * x)

    def dmp(self, deadline):
        args = {'df': 1, 'loc': 0, 'scale': 1}
        dmp_task = 0
        for k in range(self._n_components):
            dmp_task += self.weights_[k] * abs(int(deadline > self._mean(k)) - chi2.cdf(self.normalize(deadline, k), **args))
        return dmp_task

    def _complete_likelihood(self, X, zz, mode, cv):
        return sum([zz[i] * rInvGauss(mode, cv).log_pdf(x_i) for i, x_i in enumerate(X)])

    def _derivative_complete_likelihood(self, X, zz, mode, cv):
        return np.array([zz[i] * rInvGauss(mode, cv)._dlogf(x_i) for i, x_i in enumerate(X)]).sum(axis=0)

    def _second_derivative_complete_likelihood(self, X, zz, mode, cv):
        return sum([zz[i] * rInvGauss(mode, cv)._hesslogf(x_i) for i, x_i in enumerate(X)])

    def _update_weights(self, X):
        zz = np.zeros((len(X), self._n_components))
        for i, x_i in enumerate(X):
            zz[i, :] = np.array(self._proba_components(x_i)) / self.pdf(x_i)
        return zz

    def _score_complete(self, X, z):
        l1 = sum([sum([z[i, j] * log(pi_j) for j, pi_j in enumerate(self.weights_)]) for i, _ in enumerate(X)])
        l2 = sum([sum([z[i, j] * rInvGauss(self.modes_[j], self.cv_[j]).log_pdf(x_i)
                       for i, x_i in enumerate(X)]) for j in range(self._n_components)])
        return l1 + l2

    def score(self, X, y=None):
        return sum([log(self.pdf(x)) for x in X])

    def likelihood(self, X):
        return np.prod([self.pdf(x) for x in X])

    def _EM(self, X, verbose=False, method='dogleg'):
        l2 = self.score(X)
        max_iter = self.n_iter_
        l1 = 0
        l2_inf = 0
        pbar = trange(self.n_iter_) if self.verbose or verbose else range(self.n_iter_)

        for _ in pbar:
            max_iter -= 1
            # E step
            z = self._update_weights(X)
            # M step
            self._M_step(X, z, method)
            # score
            l0 = l1
            l1 = l2
            l2 = self.score(X)
            if l0 - l1 == 0:
                continue
            elif self.converged_:
                if self.verbose or verbose:
                    print('Converged in {} iterations'.format(self.n_iter_ - max_iter - 1))
                return self
            else:
                aitken_acceleration = (l2 - l1) / (l1 - l0)
                l1_inf = l2_inf
                l2_inf = l1 + (l2 - l1) / (1 - aitken_acceleration)
                self.converged_ = abs(l2_inf - l1_inf) < self.tol
            if self.verbose or verbose:
                pbar.set_description('acceleration = {}'.format(aitken_acceleration))
        if verbose or self.verbose:
            print('Not converged...')
        return self

    def fit(self, X, y=None, verbose=False, method='dogleg'):
        self.fitted_ = True
        assert all([xx > 0 for xx in X]), "non-positive value"
        XX = np.array(X).copy()
        self.initialize(XX)

        if self._n_components > 1:
            return self._EM(XX, verbose=verbose, method=method)
        elif self._n_components == 1:
            uni = rInvGauss(mode=self.modes_[0] if self.modes_ else None,
                            cv=self.cv_[0] if self.cv_ else None,
                            max_iter=self.n_iter_, tol=self.tol, verbose=self.verbose).fit(XX)
            self.modes_ = [uni.mode]
            self.cv_ = [uni.cv]
            self.weights_ = [1.]
            return self
        else:
            raise ValueError('n_component must be >= 1')

    def aic(self, X):
        return 2 * len(X) * self.score(X) - (3 * self._n_components - 1) * 2

    def bic(self, X):
        return 2 * self.score(X) - (3 * self._n_components - 1) * np.log(len(X))

    def predict_proba(self, X):
        return [self._proba_components(x) for x in X]

    def predict(self, X):
        return [np.argmax(self._proba_components(x)) for x in X]

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def sample(self, n_sample=1):
        if n_sample < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self._n_components)
            )

        # https://fr.wikipedia.org/wiki/Loi_inverse-gaussienne#Simulation_num%C3%A9rique_de_la_loi_inverse-gaussienne
        clusters_ = np.random.choice(a=range(self._n_components), p=self.weights_, size=n_sample)
        mu = np.zeros(n_sample)
        lambd = np.zeros(n_sample)
        for i, k in enumerate(clusters_):
            mu[i] = rInvGauss(mode=self.modes_[k], cv=self.cv_[k]).mean
            lambd[i] = rInvGauss(mode=self.modes_[k], cv=self.cv_[k]).shape
        y = np.random.normal(size=n_sample) ** 2
        X = mu + (mu ** 2 * y - mu * np.sqrt(4 * mu * lambd * y + mu ** 2 * y ** 2)) / (2 * lambd)
        U = np.random.rand(n_sample)
        S = np.zeros(n_sample)
        Z = mu / (mu + X)
        ok = (U <= Z)

        notok = (U > Z)
        S[ok] = X[ok]
        S[notok] = mu[notok] ** 2 / X[notok]
        return S

    def ks_test(self, sample):
        y = self.fit_predict(sample)
        for k in range(self._n_components):
            sample0 = np.array(sample)[np.array(y) == k]
            mean = rInvGauss()._mean(self.modes_[k], self.cv_[k])
            shape = rInvGauss()._shape(self.modes_[k], self.cv_[k])
            chi = shape * (sample0 - mean) ** 2 / (mean ** 2 * sample0)
            print(f'Component {k} :')
            print(kstest(chi, 'chi2', args=(1,)))

    def _update_params(self, XX, zz, x0, method='dogleg'):
        hess_LL = lambda x: -self._second_derivative_complete_likelihood(XX, zz, x[0], x[1])
        grad_LL = lambda x: -self._derivative_complete_likelihood(XX, zz, x[0], x[1])
        LL = lambda x: -self._complete_likelihood(XX, zz, x[0], x[1])
        res = minimize(fun=LL, method=method, x0=x0, jac=grad_LL, hess=hess_LL)['x']
        return res

    def initialize(self, X, method=None):
        kmeans = KMeans(self._n_components).fit(X.copy().reshape(-1, 1))

        z = np.zeros((len(X), self._n_components))
        if self.weights_ is None:
            for i, j in enumerate(kmeans.predict(X.reshape(-1, 1))):
                z[i, j] = 1
            self.weights_ = np.mean(z, axis=0).tolist()

        self.modes_ = kmeans.cluster_centers_.reshape(-1)
        if self.cv_ is None:
            self.cv_ = [1.] * self._n_components

    def _M_step(self, X, z, method):
        self.weights_ = np.mean(z, axis=0).tolist()
        for j in range(self._n_components):
            self.modes_[j], self.cv_[j] = self._update_params(X, z[:, j], method=method,
                                                              x0=np.array((self.modes_[j], self.cv_[j])))
        return 0

    def get_parameters(self):
        return {'weights': list(self.weights_), 'modes': list(self.modes_),
                'cv': list(self.cv_), 'n_components': self._n_components}

    def set_parameters(self, params):
        self.weights_ = params['weights']
        self.cv_ = params['cv']
        self.modes_ = params['modes']

