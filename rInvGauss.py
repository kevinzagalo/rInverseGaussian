import numpy
from numpy import sqrt, log, exp
from scipy.optimize import fmin_bfgs

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

    def kde(self, X, gamma=None):
        if gamma:
            pass
        else:
            gamma = self.min_lcv(X)
        return lambda t: numpy.mean([self.pdf(t, x, gamma) for x in X])

    def lcv(self, X, gamma):
        X = list(X)
        return numpy.mean([log(self.kde(X[:i]+X[i+1:], gamma)(x)) for i, x in enumerate(X)])

    def min_lcv(self, X):
        gamma = fmin_bfgs(f=lambda g: self.lcv(X, g), x0=numpy.array([1.0]), disp=False)
        return max((gamma[0], 0.001))

    def sample(self, n_sample=1):
        if n_sample < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at least one sample." % (n_sample)
            )

        # https://fr.wikipedia.org/wiki/Loi_inverse-gaussienne#Simulation_num%C3%A9rique_de_la_loi_inverse-gaussienne
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    t_range = numpy.linspace(0.1, 25)
    rIG = rInvGauss(gamma=4.0)
    print(rIG.get_parameters())
    sample = rIG.sample(1000)
    plt.hist(sample, density=True, bins=50)
    plt.plot(t_range, rIG.pdf(t_range))
    plt.ylim(0, 0.8)
    plt.title('A generated sample')
    plt.show()

    gamma_range = numpy.linspace(0.1, 5, 10)
    lcv = [rIG.lcv(sample, g) for g in gamma_range]
    plt.plot(gamma_range, lcv)
    plt.show()

    for f in os.listdir('./data'):
        if '.csv' not in f:
            continue
        df_rtime = pd.read_csv('./data/'+f)
        sample = df_rtime.rtime

        #model = rInvGaussMixture(2).fit(sample)
        #print(model.get_parameters())

        t_range = numpy.linspace(1, max(sample))
        plt.hist(sample, bins=75, density=True)
        #kernel_t_range = [rInvGauss().kde(sample)(tt) for tt in t_range]
        #plt.plot(t_range, kernel_t_range)
        plt.title(f[:-4])
        plt.show()