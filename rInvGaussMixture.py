import numpy as np
from math import sqrt, log, exp
from scipy.optimize import minimize, fmin_bfgs
from tqdm import trange
from rInvGauss import rInvGauss
from sklearn.cluster import KMeans


class rInvGaussMixture:

    def __init__(self, n_components=1, max_iter=100, tol=1e-4, modes_init=None, shapes_init=None, weights_init=None):
        self.tol = tol,
        self.n_iter_ = max_iter

        if weights_init:
            assert len(weights_init) == n_components, 'Weights lengths should be equal to n_components'
        else:
            weights_init = [1./n_components] * n_components

        self._n_components = n_components
        self.modes_ = modes_init
        self.shapes_ = shapes_init
        self.weights_ = weights_init
        self.converged_ = False

    def _proba_components(self, x):
        return [pi_j * rInvGauss(self.modes_[j], self.shapes_[j]).pdf(x) for j, pi_j in enumerate(self.weights_)]

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

    def _update_params(self, XX, zz, x0):
        hess_LL = lambda x: -self._second_derivative_complete_likelihood(XX, zz, x[0], x[1])
        grad_LL = lambda x: -self._derivative_complete_likelihood(XX, zz, x[0], x[1])
        LL = lambda x: -self._complete_likelihood(XX, zz, x[0], x[1])
        res = minimize(fun=LL, method='dogleg', x0=x0, jac=grad_LL, hess=hess_LL)
        return res['x']

    def _score_complete(self, X, z):
        l1 = sum([sum([z[i, j] * log(pi_j) for j, pi_j in enumerate(self.weights_)]) for i, _ in enumerate(X)])
        l2 = sum([sum([z[i, j] * rInvGauss(self.modes_[j], self.shapes_[j]).log_pdf(x_i)
                       for i, x_i in enumerate(X)]) for j in range(self._n_components)])
        return l1 + l2

    def score(self, X, y=None):
        return sum([log(self.pdf(x)) for x in X])

    def _EM(self, XX, verbose=False):
        X = np.array(XX).copy()
        kmeans = KMeans(self._n_components).fit(X.reshape(-1, 1))
        self.modes_ = kmeans.cluster_centers_.reshape(-1)
        self.shapes_ = [1.] * self._n_components
        z = np.zeros((len(X), self._n_components))
        for i, j in enumerate(kmeans.predict(X.reshape(-1, 1))):
            z[i, j] = 1
        self.weights_ = np.mean(z, axis=0).tolist()
        likelihood = self._score_complete(X, z)
        max_iter = self.n_iter_
        old_l = 0

        for _ in trange(self.n_iter_):
            max_iter -= 1
            old_likelihood = old_l
            old_l = likelihood

            # E-step
            z = self._update_weights(X)

            # M-step
            self.weights_ = np.mean(z, axis=0).tolist()
            for j in range(self._n_components):
                self.modes_[j], self.shapes_[j] = self._update_params(X, z[:, j],
                                                                      np.array((self.modes_[j], self.shapes_[j])))

            # score
            likelihood = self._score_complete(X, z)
            aitken_acceleration = (likelihood - old_l) / (old_l - old_likelihood)
            self.converged_ = abs((likelihood - old_l)/(1-aitken_acceleration)) < self.tol
            if self.converged_:
                print('Converged in {} iterations'.format(self.n_iter_ - max_iter + 1))
                return self
        print('Not converged...')
        return self

    def fit(self, X, y=None, verbose=False, method='EM'):
        return self._EM(X, verbose=verbose)

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
            mu[i] = rInvGauss(theta=self.modes_[k], gamma=self.shapes_[k]).mu
            lambd[i] = rInvGauss(theta=self.modes_[k], gamma=self.shapes_[k]).lambd
        y = np.random.normal(size=n_sample)**2
        X = mu + (mu**2 * y - mu * np.sqrt(4 * mu * lambd * y +mu**2 * y**2)) / (2 * lambd)
        U = np.random.rand(n_sample)
        S = np.zeros(n_sample)
        Z = mu / (mu + X)
        ok = (U <= Z)
        notok = (U > Z)
        S[ok] = X[ok]
        S[notok] = mu[notok]**2 / X[notok]
        return S

    def get_parameters(self):
        return {'weights': self.weights_, 'modes': self.modes_,
                'shapes': self.shapes_, 'n_components': self._n_components}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    #for f in os.listdir('data'):
    #    if 'xtimes' in f:
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
#
    #    t_range = np.linspace(1, max(x))
    #    plt.hist(x, density=True, bins=50)
    #    plt.plot(t_range, rIG.pdf(t_range))
    #    plt.title(f[:-6])
    #    plt.show()

    x = pd.read_csv('data/actl_5.csv').values[:, 1]

    BICS = []
    AICS = []
    t_range = np.linspace(1, max(x))
    plt.hist(x, density=True)
    for n_components in range(2, 10):
        rIG = rInvGaussMixture(n_components).fit(x)
        plt.plot(t_range, rIG.pdf(t_range), label='n_component={}'.format(n_components), ls='dotted')
        BICS.append(rIG.bic(x))
        AICS.append(rIG.aic(x))
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(range(2, 10), BICS, label='BIC')
    ax_ = ax.twinx()
    ax.set_ylabel('BIC')
    plt.xlabel('m')
    ax_.plot(range(2, 10), AICS, label='AIC')
    ax_.set_ylabel('AIC')
    plt.legend()
    plt.show()

