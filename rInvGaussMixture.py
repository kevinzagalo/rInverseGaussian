import numpy
from math import sqrt, log, exp
from scipy.optimize import minimize, fmin_bfgs
from tqdm import tqdm

class rInvGauss:

    def __init__(self, theta=1.0, gamma=1.0):
        self.theta = theta
        self.gamma = gamma

    def _mu(self, theta, gamma):
        return sqrt(theta * (3 * gamma + theta))

    def _lambd(self, theta, gamma):
        return theta * (3 * gamma + theta) / gamma

    @property
    def mu(self):
        return self._mu(self.theta, self.gamma)

    @property
    def lambd(self):
        return self._lambd(self.theta, self.gamma)

    def pdf(self, x, theta=None, gamma=None):
        assert x > 0, 'x must be non-negative'
        if theta and gamma:  # In case we want to use only the pdf without the object parameters
            mu = self._mu(theta, gamma)
            lambd = self._lambd(theta, gamma)
        else:  # in case we use the object parameters
            mu = self.mu
            lambd = self.lambd
        a1 = sqrt(lambd / (2 * numpy.pi * x**3))
        a2 = lambd * (x - mu)**2 / (x * mu**2)
        return a1 * exp(-a2/2)

    def log_pdf(self, x):
        assert x > 0, 'x must be non-negative'
        print(self.lambd)
        a1 = log(self.lambd)/2 - log(2) - log(numpy.pi) - 3 * log(x)
        a2 = self.lambd * (x - self.mu)**2 / (x * self.mu**2)
        return a1 - a2/2

    def _dlogf(self, x):
        p = 3 * self.gamma + self.theta
        dLL_dtheta = - 3 / (2 * x) \
                     - self.theta / (x * self.gamma) \
                     + 1 / p + 3 * self.gamma / (2 * self.theta * p) \
                     + sqrt(self.theta / p) / (2 * self.gamma) \
                     + sqrt(p / self.theta) / (2 * self.gamma)
        dLL_dgamma = x / (2 * self.gamma ** 2) \
                     + self.theta ** 2 / (2 * x * self.gamma ** 2) \
                     - self.theta / (2 * self.gamma * p) \
                     + 3 * sqrt(self.theta / p) / (2 * self.gamma) \
                     - sqrt(self.theta * p) / self.gamma ** 2
        return numpy.array([dLL_dtheta, dLL_dgamma])

    def _hesslogf(self, x):
        p = 3 * self.gamma + self.theta
        dLL_dtheta2 = -0.25 * (4 / (x * self.gamma) + 2 / self.theta ** 2 + 2 / p ** 2 + 9 * self.gamma / sqrt(self.theta * p) ** 3)
        dLL_dgamma2 = -x / self.gamma ** 3 - self.theta ** 2 / (x * self.gamma ** 3) \
                      + 3 * self.theta / (2 * self.gamma * p ** 2) \
                      - 9 * sqrt(self.theta) / (4 * self.gamma * sqrt(p ** 3)) \
                      + self.theta / (2 * self.gamma ** 2 * p) \
                      - 3 * sqrt(self.theta) / (self.gamma ** 2 * sqrt(p)) \
                      + 2 * sqrt(self.theta * p) / self.gamma ** 3
        dLL_dtheta_dgamma = self.theta / (x * self.gamma ** 2) \
                            - (27 * self.gamma ** 3 + 30 * self.gamma * self.theta ** 2 \
                               + 4 * self.theta ** 3 \
                               + 3 * self.gamma ** 2 * (21 * self.theta + 2 * sqrt(self.theta * p))) \
                            / (4 * self.gamma ** 2 * sqrt(self.theta * p ** 5))
        return numpy.matrix([[dLL_dtheta2, dLL_dtheta_dgamma], [dLL_dtheta_dgamma, dLL_dgamma2]])

    def kde(self, X):
        return lambda t: numpy.mean([self.pdf(t, x, self.gamma) for x in X])

    def sample(self, n_sample=1):
        y = numpy.random.normal(size=n_sample)**2
        X = self.mu + (self.mu**2 * y - self.mu * numpy.sqrt(4 * self.mu * self.lambd * y + self.mu**2 * y**2)) / (2 * self.lambd)
        U = numpy.random.rand(n_sample)
        S = numpy.zeros(n_sample)
        Z = self.mu / (self.mu + X)
        ok = (U <= Z)
        notok = (U > Z)
        S[ok] = X[ok]
        S[notok] = self.mu**2 / X[notok]
        return S

    def get_parameters(self):
        return {'mode': self.theta, 'shape': self.gamma}


class rInvGaussMixture:

    def __init__(self, n_components=1, max_iter=100, tol=1e-4, modes_init=1., shapes_init=1., weights_init=None):
        self.tol = tol,
        self.n_iter_ = max_iter

        if isinstance(n_components, int):
            self._n_components = n_components
        if isinstance(modes_init, int) or isinstance(modes_init, float):
            modes_init = [modes_init]
        if isinstance(shapes_init, int) or isinstance(shapes_init, float):
            shapes_init = [shapes_init]
        if weights_init:
            assert len(weights_init) == n_components, 'Weights lenghts should be equal to n_components'
        else:
            weights_init = [1./n_components] * n_components

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
        return numpy.array([zz[i] * rInvGauss(theta, gamma)._dlogf(x_i) for i, x_i in enumerate(X)]).sum(axis=0)

    def _second_derivative_complete_likelihood(self, X, zz, theta, gamma):
        return sum([zz[i] * rInvGauss(theta, gamma)._hesslogf(x_i) for i, x_i in enumerate(X)])

    def _update_weights(self, X):
        zz = numpy.zeros((len(X), self._n_components))
        for i, x_i in enumerate(X):
            zz[i, :] = numpy.array(self._proba_components(x_i)) / self.pdf(x_i)
        return zz

    def _update_params(self, XX, zz, x0):
        hess_LL = lambda x: -self._second_derivative_complete_likelihood(XX, zz, x[0], x[1])
        grad_LL = lambda x: -self._derivative_complete_likelihood(XX, zz, x[0], x[1])
        LL = lambda x: -self._complete_likelihood(XX, zz, x[0], x[1])
        res = minimize(fun=LL, method='Newton-CG', x0=x0, jac=grad_LL, hess=hess_LL)
        #return fmin_bfgs(f=LL, x0=x0, fprime=grad_LL, disp=False)
        return res['x']

    def _score_complete(self, X, z):
        l1 = sum([sum([z[i, j] * log(pi_j) for j, pi_j in enumerate(self.weights_)]) for i, _ in enumerate(X)])
        l2 = sum([sum([z[i, j] * rInvGauss(self.modes_[j], self.shapes_[j]).log_pdf(x_i)
                       for i, x_i in enumerate(X)]) for j in range(self._n_components)])
        return l1 + l2

    def score_sample(self, X):
        return [sum([self.weights_[j] * rInvGauss(self.modes_[j], self.shapes_[j]).pdf(x_i)
                     for j in range(self._n_components)]) for i, x_i in enumerate(X)]

    def score(self, X, y=None):
        return sum(self.score_sample(X))

    def _EM(self, X, verbose=False):
        self.weights_ = [1/self._n_components] * self._n_components
        self.modes_ = [1.] * self._n_components
        self.shapes_ = [1.] * self._n_components
        likelihood = self.score(X)
        max_iter = self.n_iter_
        old_l = 0

        for _ in tqdm(range(self.n_iter_), ascii=True, desc='iterations'):
            max_iter -= 1
            old_likelihood = old_l
            old_l = likelihood

            # E-step
            z = self._update_weights(X)

            # M-step
            self.weights_ = numpy.mean(z, axis=0).tolist()
            for j in range(self._n_components):
                self.modes_[j], self.shapes_[j] = self._update_params(X, z[:, j],
                                                                      numpy.array((self.modes_[j], self.shapes_[j])))

            likelihood = self._score_complete(X, z)
            aitken_acceleration = (likelihood - old_l) / (old_l - old_likelihood)
            self.converged_ = abs((likelihood - old_l)/(1-aitken_acceleration)) < self.tol
            if self.converged_:
                print('Converged in {} iterations'.format(self.n_iter_ - max_iter))
                return self

        print('Not converged...')
        return self

    def fit(self, X, y=None, verbose=False):
        return self._EM(X, verbose=verbose)

    def aic(self, X):
        return 2 * len(X) * self.score(X) - (3 * self._n_components - 1) * 2

    def bic(self, X):
        return 2 * len(X) * self.score(X) - (3 * self._n_components - 1) * numpy.log(len(X))

    def predict_proba(self, X):
        return [self._proba_components(x) for x in X]

    def predict(self, X):
        return [numpy.argmax(self._proba_components(x)) for x in X]

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def sample(self, n_sample=1):
        if n_sample < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        # https://fr.wikipedia.org/wiki/Loi_inverse-gaussienne#Simulation_num%C3%A9rique_de_la_loi_inverse-gaussienne
        clusters_ = numpy.random.choice(a=range(self._n_components), p=self.weights_, size=n_sample)
        mu = numpy.zeros(n_sample)
        lambd = numpy.zeros(n_sample)
        for i, k in enumerate(clusters_):
            mu[i] = rInvGauss()._mu(self.modes_[k], self.shapes_[k])
            lambd[i] = rInvGauss()._lambd(self.modes_[k], self.shapes_[k])
        y = numpy.random.normal(size=n_sample)**2
        X = mu + (mu**2 * y - mu * numpy.sqrt(4 * mu * lambd * y +mu**2 * y**2)) / (2 * lambd)
        U = numpy.random.rand(n_sample)
        S = numpy.zeros(n_sample)
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

    for f in os.listdir('./data'):
        if '.csv' not in f:
            continue
        df_rtime = pd.read_csv('./data/'+f)
        sample = df_rtime.rtime

        #model = rInvGaussMixture(2).fit(sample)
        #print(model.get_parameters())

        t_range = numpy.linspace(1, max(sample))
        plt.hist(sample, bins=75, density=True)
        kernel_t_range = [rInvGauss().kde(sample)(tt) for tt in t_range]

        plt.plot(t_range, kernel_t_range)
        plt.title(f[:-4])
        plt.show()
