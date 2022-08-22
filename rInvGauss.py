import numpy
from numpy import sqrt, log, exp
from scipy.optimize import minimize
from tqdm import trange


class rInvGauss:

    def __init__(self, theta=None, gamma=None, tol=1e-4, max_iter=100, verbose=False):
        self.theta = theta
        self.gamma = gamma
        self.n_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _mu(self, theta, gamma):
        return sqrt(theta * (3 * gamma + theta))

    def _lambd(self, theta, gamma):
        return theta * (3 * gamma + theta) / gamma

    def _theta(self, mu, lambd):
        return mu * (sqrt(1 + (1.5 * mu / lambd) ** 2) - 1.5 * mu / lambd)

    def _gamma(self, mu, lambd):
        return mu ** 2 / lambd

    def _checkvalues(self):
        if self.theta <= 0 or self.gamma <= 0:
            raise ValueError(
                'theta = {} and gamma = {} must be positive'.format(self.theta, self.gamma)
            )

    @property
    def mu(self):
        return self._mu(self.theta, self.gamma)

    @property
    def lambd(self):
        return self._lambd(self.theta, self.gamma)

    def pdf(self, x, theta=None, gamma=None):
        self._checkvalues()
        if theta and gamma:  # In case we want to use only the pdf without the object parameters
            mu = self._mu(theta, gamma)
            lambd = self._lambd(theta, gamma)
        else:  # in case we use the object parameters
            mu = self.mu
            lambd = self.lambd
        a1 = sqrt(lambd / (2 * numpy.pi * x ** 3))
        a2 = lambd * (x - mu) ** 2 / (x * mu ** 2)
        return a1 * exp(-a2 / 2)

    def log_pdf(self, x, theta=None, gamma=None):
        self._checkvalues()
        if theta and gamma:  # In case we want to use only the pdf without the object parameters
            mu = self._mu(theta, gamma)
            lambd = self._lambd(theta, gamma)
        else:  # in case we use the object parameters
            mu = self.mu
            lambd = self.lambd
        a1 = log(lambd) - log(2) - log(numpy.pi) - 3 * log(x)
        a2 = lambd * (x - mu) ** 2 / (x * mu ** 2)
        return a1 / 2 - a2 / 2

    def _dlogf(self, x, theta=None, gamma=None):
        self._checkvalues()
        if theta and gamma:
            pass
        else:
            theta = self.theta
            gamma = self.gamma
        p = 3 * gamma + theta
        dLL_dtheta = - 1.5 / x \
                     - theta / (x * gamma) \
                     + 1 / p \
                     + 1.5 * gamma / (theta * p) \
                     + sqrt(theta) / (2 * gamma * sqrt(p)) \
                     + sqrt(p / theta) / (2 * gamma)
        dLL_dgamma = x / (2 * gamma ** 2) \
                     + theta ** 2 / (2 * x * gamma ** 2) \
                     - theta / (2 * gamma * p) \
                     + 1.5 * sqrt(theta) / (gamma * sqrt(p)) \
                     - sqrt(theta * p) / gamma ** 2
        return numpy.array([dLL_dtheta, dLL_dgamma])

    def _hesslogf(self, x, theta=None, gamma=None):
        self._checkvalues()
        if theta and gamma:
            pass
        else:
            theta = self.theta
            gamma = self.gamma
        p = 3 * gamma + theta
        dLL_dtheta2 = -0.25 * (4 / (x * gamma) + 2 / (theta ** 2) + 2 / (p ** 2) + 9 * gamma / sqrt(theta * p) ** 3)
        dLL_dgamma2 = -x / (gamma ** 3) \
                      - theta ** 2 / (x * gamma ** 3) \
                      + 1.5 * theta / (gamma * p ** 2) \
                      - 9 * sqrt(theta) / (4 * gamma * sqrt(p) ** 3) \
                      + theta / (2 * gamma ** 2 * p) \
                      - 3 * sqrt(theta) / (gamma ** 2 * sqrt(p)) \
                      + 2 * sqrt(theta * p) / gamma ** 3
        dLL_dtheta_dgamma = theta / (x * gamma ** 2) \
                            - (27 * gamma ** 3 + 30 * gamma * theta ** 2 \
                               + 4 * theta ** 3 \
                               + 3 * gamma ** 2 * (21 * theta + 2 * sqrt(theta * p))) \
                            / (4 * gamma ** 2 * sqrt(theta * p ** 5))
        return numpy.matrix([[dLL_dtheta2, dLL_dtheta_dgamma], [dLL_dtheta_dgamma, dLL_dgamma2]])

    def score(self, X, y=None, theta=None, gamma=None):
        return sum([self.log_pdf(x, theta, gamma) for x in X])

    def _update_params(self, XX, x0):
        hess_LL = lambda y: sum([-self._hesslogf(x, y[0], y[1]) for x in XX])
        grad_LL = lambda y: numpy.array([-self._dlogf(x, y[0], y[1]) for x in XX]).sum(axis=0)
        LL = lambda y: -self.score(XX, theta=y[0], gamma=y[1])
        res = minimize(fun=LL, method='dogleg', x0=x0, jac=grad_LL, hess=hess_LL)['x']
        return res

    def fit(self, XX):
        X = numpy.array(XX).copy()

        if self.theta is None:
            self.theta = numpy.mean(X)
        if self.gamma is None:
            self.gamma = 1.

        l2 = self.score(X)
        max_iter = self.n_iter
        l1 = 0
        l2_inf = 0
        pbar = trange(self.n_iter)
        for _ in pbar:
            l0 = l1
            l1 = l2
            self.theta, self.gamma = self._update_params(X, numpy.array((self.theta, self.gamma)))
            l2 = self.score(X)
            aitken_acceleration = (l2 - l1) / (l1 - l0)
            l1_inf = l2_inf
            l2_inf = l1 + (l2 - l1) / (1 - aitken_acceleration)
            self.converged_ = abs(l2_inf - l1_inf) / (1 - aitken_acceleration) < self.tol
            pbar.set_description('acceleration = {}'.format(aitken_acceleration))
            if self.converged_:
                if self.verbose:
                    print('Converged in {} iterations'.format(self._n_iter - max_iter + 1))
                return self
            max_iter -= 1
        print('Not converged...')
        return self

    def kde(self, X, gamma=None):
        if gamma:
            pass
        else:
            gamma = self.gamma
        return lambda t: numpy.mean([self.pdf(t, x, gamma) for x in X])

    def sample(self, n_sample=1):
        if n_sample < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at least one sample." % (n_sample)
            )

        # https://fr.wikipedia.org/wiki/Loi_inverse-gaussienne#Simulation_num%C3%A9rique_de_la_loi_inverse-gaussienne
        y = numpy.random.normal(size=n_sample) ** 2
        X = self.mu + (
                    self.mu ** 2 * y - self.mu * numpy.sqrt(4 * self.mu * self.lambd * y + self.mu ** 2 * y ** 2)) / (
                        2 * self.lambd)
        U = numpy.random.rand(n_sample)
        S = numpy.zeros(n_sample)
        Z = self.mu / (self.mu + X)
        ok = (U <= Z)
        notok = (U > Z)
        S[ok] = X[ok]
        S[notok] = self.mu ** 2 / X[notok]
        return S

    def get_parameters(self):
        return {'mode': self.theta, 'shape': self.gamma}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from scipy.stats import invgauss

    sample = rInvGauss(theta=10, gamma=4.0).sample(1000)

    rIG1 = rInvGauss(gamma=4.0).fit(sample)
    rIG2 = rInvGauss(max_iter=500).fit(sample)

    print(rIG1.get_parameters())
    print(rIG2.get_parameters())

    plt.hist(sample, density=True, bins=50, color='black')
    t_range = numpy.linspace(0.1, max(sample))
    plt.plot(t_range, rIG1.pdf(t_range), color='red', label='gamma fixed')
    plt.plot(t_range, rIG2.pdf(t_range), color='blue', label='gamma not fixed')
    # plt.ylim(0, 0.8)
    plt.legend()
    plt.title('A generated sample with MLE')
    plt.show()
    # plt.hist(sample, density=True, bins=50)
    # t_range = numpy.linspace(0.1, max(sample))
#
# kernel_t_range = [rIG.kde(sample)(tt) for tt in t_range]
# plt.plot(t_range, kernel_t_range, color='red')
# plt.ylim(0, 0.8)
# plt.title('A generated sample with kde')
# plt.show()
