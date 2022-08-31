import numpy as np
from math import sqrt, log, exp
from scipy.optimize import minimize, fmin_bfgs
from tqdm import trange
from rInvGauss import rInvGauss
from sklearn.cluster import KMeans
import abc

class rInvGaussMixtureCore:

    def __init__(self, n_components, max_iter=100, tol=1e-4, modes_init=None,
                 smooth_init=None, weights_init=None, verbose=False):
        self.tol = tol
        self.n_iter_ = max_iter
        self.verbose = verbose
        self._n_components = n_components

        if weights_init is not None:
            assert len(weights_init) == self._n_components, 'Weights lengths should be equal to n_components'

        if modes_init is not None:
            assert len(modes_init) == self._n_components, 'Modes lengths should be equal to n_components'

        self.modes_ = modes_init
        self.smooth_ = smooth_init
        self.weights_ = weights_init
        self.converged_ = False

    def _proba_components(self, x):
        return [pi_j * rInvGauss(self.modes_[j], self.smooth_[j]).pdf(x) for j, pi_j in enumerate(self.weights_)]

    def pdf(self, x):
        return sum(self._proba_components(x))

    def _complete_likelihood(self, X, zz, theta, gamma):
        return sum([zz[i] * rInvGauss(theta, gamma).log_pdf(x_i) for i, x_i in enumerate(X)])

    def _derivative_complete_likelihood(self, X, zz, theta, gamma):
        return np.array([zz[i] * rInvGauss(theta, gamma)._dlogf(x_i) for i, x_i in enumerate(X)]).sum(axis=0)

    def _second_derivative_complete_likelihood(self, X, zz, theta, gamma):
        return sum([zz[i] * rInvGauss(theta, gamma)._hesslogf(x_i) for i, x_i in enumerate(X)])

    def _update_weights(self, X):
        zz = np.zeros((len(X), self._n_components))
        for i, x_i in enumerate(X):
            zz[i, :] = np.array(self._proba_components(x_i)) / self.pdf(x_i)
        return zz

    @abc.abstractmethod
    def _update_params(self, XX, zz, x0):
        pass

    def _score_complete(self, X, z):
        l1 = sum([sum([z[i, j] * log(pi_j) for j, pi_j in enumerate(self.weights_)]) for i, _ in enumerate(X)])
        l2 = sum([sum([z[i, j] * rInvGauss(self.modes_[j], self.smooth_[j]).log_pdf(x_i)
                       for i, x_i in enumerate(X)]) for j in range(self._n_components)])
        return l1 + l2

    def score(self, X, y=None):
        return sum([log(self.pdf(x)) for x in X])

    @abc.abstractmethod
    def initialize(self, X, method='kmeans'):
        pass

    @abc.abstractmethod
    def _M_step(self, X, z, method):
        pass

    def _EM(self, XX, verbose=False, method='dogleg'):
        assert all([xx > 0 for xx in XX]), "non-positive value"
        X = np.array(XX).copy()
        self.initialize(X)

        if self._n_components > 1:
            l2 = self.score(X)
            max_iter = self.n_iter_
            l1 = 0
            l2_inf = 0
            pbar = trange(self.n_iter_)
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
                aitken_acceleration = (l2 - l1) / (l1 - l0)
                l1_inf = l2_inf
                l2_inf = l1 + (l2 - l1) / (1 - aitken_acceleration)
                self.converged_ = abs(l2_inf - l1_inf) < self.tol
                if self.converged_:
                    if self.verbose or verbose:
                        print('Converged in {} iterations'.format(self.n_iter_ - max_iter - 1))
                    return self
                pbar.set_description('acceleration = {}'.format(aitken_acceleration))
            print('Not converged...')
        elif self._n_components == 1:
            uni = rInvGauss(theta=self.modes_[0] if self.modes_ else None,
                            gamma=self.smooth_[0] if self.smooth_ else None,
                            max_iter=self.n_iter_, tol=self.tol, verbose=self.verbose).fit(X)
            self.modes_ = [uni.theta]
            self.smooth_ = [uni.gamma]
            self.weights_ = [1.]
        return self

    def fit(self, X, y=None, verbose=False, method='dogleg'):
        return self._EM(X, verbose=verbose, method=method)

    def aic(self, X):
        return 2 * len(X) * self.score(X) - (3 * self._n_components - 1) * 2

    def bic(self, X):
        return 2 * len(X) * self.score(X) - (3 * self._n_components - 1) * np.log(len(X))

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
                "least one sample." % (self.n_components)
            )

        # https://fr.wikipedia.org/wiki/Loi_inverse-gaussienne#Simulation_num%C3%A9rique_de_la_loi_inverse-gaussienne
        clusters_ = np.random.choice(a=range(self._n_components), p=self.weights_, size=n_sample)
        mu = np.zeros(n_sample)
        lambd = np.zeros(n_sample)
        for i, k in enumerate(clusters_):
            mu[i] = rInvGauss(theta=self.modes_[k], gamma=self.smooth_[k]).mu
            lambd[i] = rInvGauss(theta=self.modes_[k], gamma=self.smooth_[k]).lambd
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

    @abc.abstractmethod
    def get_parameters(self):
        pass


class rInvGaussMixture(rInvGaussMixtureCore):
    def __init__(self, n_components, max_iter=100, tol=1e-4, modes_init=None,
                 smooth_init=None, weights_init=None, verbose=False):
        super().__init__(n_components=n_components, tol=tol, max_iter=max_iter, modes_init=modes_init,
                         weights_init=weights_init, verbose=verbose, smooth_init=smooth_init)

    def _update_params(self, XX, zz, x0, method='dogleg'):
        hess_LL = lambda x: -self._second_derivative_complete_likelihood(XX, zz, x[0], x[1])
        grad_LL = lambda x: -self._derivative_complete_likelihood(XX, zz, x[0], x[1])
        LL = lambda x: -self._complete_likelihood(XX, zz, x[0], x[1])
        res = minimize(fun=LL, method=method, x0=x0, jac=grad_LL, hess=hess_LL)['x']
        return res

    def initialize(self, X, method=None):
        kmeans = KMeans(self._n_components).fit(X.reshape(-1, 1))

        z = np.zeros((len(X), self._n_components))
        if self.weights_ is None:
            for i, j in enumerate(kmeans.predict(X.reshape(-1, 1))):
                z[i, j] = 1
            self.weights_ = np.mean(z, axis=0).tolist()

        self.modes_ = kmeans.cluster_centers_.reshape(-1)
        if self.smooth_ is None:
            self.smooth_ = [1.] * self._n_components

    def _M_step(self, X, z, method):
        self.weights_ = np.mean(z, axis=0).tolist()
        for j in range(self._n_components):
            self.modes_[j], self.smooth_[j] = self._update_params(X, z[:, j], method=method,
                                                                  x0=np.array((self.modes_[j], self.smooth_[j])))
        return 0

    def get_parameters(self):
        return {'weights': self.weights_, 'modes': self.modes_,
                'smooth': self.smooth_, 'n_components': self._n_components}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    sample = rInvGaussMixture(n_components=2, weights_init=[0.3, 0.7], modes_init=[10, 100],
                              smooth_init=[1, 4.0]).sample(1000)

    rIG1 = rInvGaussMixture(n_components=2, smooth_init=[1, 4.0]).fit(sample)
    rIG2 = rInvGaussMixture(n_components=2).fit(sample)

    print(rIG1.get_parameters())
    print(rIG2.get_parameters())

    plt.hist(sample, density=True, bins=50, color='black')
    t_range = np.linspace(0.1, max(sample))
    plt.plot(t_range, rIG1.pdf(t_range), color='red', label='gamma fixed')
    plt.plot(t_range, rIG2.pdf(t_range), color='blue', label='gamma not fixed')
    # plt.ylim(0, 0.8)
    plt.legend()
    plt.title('A generated sample with MLE')
    plt.show()


    # for f in os.listdir('data'):
    #    if 'xtimes' in f or 'pdf' in f:
    #        continue
    #    print(f)
    #    n_components = int(f[-5])
    #    x = pd.read_csv('data/{}'.format(f)).values[:, 1]
    #    if all(x):
    #        pass
    #    else:
    #        continue
    #    rIG = rInvGaussMixture(n_components).fit(x)
    #    print(rIG.get_parameters())
    #    t_range = np.linspace(1, max(x), 1000)
    #    plt.hist(x, density=True, bins=50, color='black')
    #    plt.plot(t_range, rIG.pdf(t_range), color='red')
    #    plt.savefig('data/{}.pdf'.format(f[:-6]))
    #    plt.show()

    # x = pd.read_csv('data/actl_5.csv').values[:, 1]
#
# BICS = []
# AICS = []
# t_range = np.linspace(1, max(x))
# plt.hist(x, density=True)
# for n_components in range(2, 10):
#    rIG = rInvGaussMixture(n_components).fit(x)
#    plt.plot(t_range, rIG.pdf(t_range), label='n_component={}'.format(n_components), ls='dotted')
#    BICS.append(rIG.bic(x))
#    AICS.append(rIG.aic(x))
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(range(2, 10), BICS, label='BIC')
# ax_ = ax.twinx()
# ax.set_ylabel('BIC')
# plt.xlabel('m')
# ax_.plot(range(2, 10), AICS, label='AIC')
# ax_.set_ylabel('AIC')
# plt.legend()
# plt.show()

