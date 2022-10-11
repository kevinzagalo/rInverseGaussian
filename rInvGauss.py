import numpy
from numpy import sqrt, log, exp
from scipy.optimize import minimize
from tqdm import trange


class rInvGauss:

    def __init__(self, mode=None, cv=None, tol=1e-4, max_iter=100, verbose=False):
        self.mode = mode
        self.cv = cv
        self.n_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.converged_ = False

    def _mean(self, mode, cv):
        return sqrt(mode * (3 * cv + mode))

    def _shape(self, mode, cv):
        return mode * (3 * cv + mode) / cv

    def _mode(self, mean, shape):
        return mean * (sqrt(1 + (1.5 * mean / shape) ** 2) - 1.5 * mean / shape)

    def _cv(self, mean, shape):
        return mean ** 2 / shape

    @property
    def mean(self):
        return self._mean(self.mode, self.cv)

    @property
    def shape(self):
        return self._shape(self.mode, self.cv)

    def pdf(self, x, mode=None, cv=None):
        if mode and cv:  # In case we want to use only the pdf without the object parameters
            mu = self._mean(mode, cv)
            lambd = self._shape(mode, cv)
        else:  # in case we use the object parameters
            mu = self.mean
            lambd = self.shape
        a1 = sqrt(lambd / (2 * numpy.pi * x ** 3))
        a2 = lambd * (x - mu) ** 2 / (x * mu ** 2)
        return a1 * exp(-a2 / 2)

    def log_pdf(self, x, mode=None, cv=None):
        if mode and cv:  # In case we want to use only the pdf without the object parameters
            mu = self._mean(mode, cv)
            lambd = self._shape(mode, cv)
        else:  # in case we use the object parameters
            mu = self.mean
            lambd = self.shape
        a1 = log(lambd) - log(2) - log(numpy.pi) - 3 * log(x)
        a2 = lambd * (x - mu) ** 2 / (x * mu ** 2)
        return a1 / 2 - a2 / 2

    def _dlogf(self, x, mode=None, cv=None):
        if mode and cv:
            pass
        else:
            mode = self.mode
            cv = self.cv
        p = 3 * cv + mode
        dLL_dmu = - 1.5 / x \
                     - mode / (x * cv) \
                     + 1 / p \
                     + 1.5 * cv / (mode * p) \
                     + sqrt(mode) / (2 * cv * sqrt(p)) \
                     + sqrt(p / mode) / (2 * cv)
        dLL_dgamma = x / (2 * cv ** 2) \
                     + mode ** 2 / (2 * x * cv ** 2) \
                     - mode / (2 * cv * p) \
                     + 1.5 * sqrt(mode) / (cv * sqrt(p)) \
                     - sqrt(mode * p) / cv ** 2
        return numpy.array([dLL_dmu, dLL_dgamma])

    def _hesslogf(self, x, mode=None, cv=None):
        if mode and cv:
            pass
        else:
            mode = self.mode
            cv = self.cv
        p = 3 * cv + mode
        dLL_dmu2 = -0.25 * (4 / (x * cv) + 2 / (mode ** 2) + 2 / (p ** 2) + 9 * cv / sqrt(mode * p) ** 3)
        dLL_dgamma2 = -x / (cv ** 3) \
                      - mode ** 2 / (x * cv ** 3) \
                      + 1.5 * mode / (cv * p ** 2) \
                      - 9 * sqrt(mode) / (4 * cv * sqrt(p) ** 3) \
                      + mode / (2 * cv ** 2 * p) \
                      - 3 * sqrt(mode) / (cv ** 2 * sqrt(p)) \
                      + 2 * sqrt(mode * p) / cv ** 3
        dLL_dmu_dgamma = mode / (x * cv ** 2) \
                            - (27 * cv ** 3 + 30 * cv * mode ** 2 \
                               + 4 * mode ** 3 \
                               + 3 * cv ** 2 * (21 * mode + 2 * sqrt(mode * p))) \
                            / (4 * cv ** 2 * sqrt(mode * p ** 5))
        try:
            return numpy.matrix([[dLL_dmu2, dLL_dmu_dgamma], [dLL_dmu_dgamma, dLL_dgamma2]])
        except:
            print(mode, cv)

    def score(self, X, y=None, mode=None, cv=None):
        return sum([self.log_pdf(x, mode, cv) for x in X])

    def _update_params(self, XX, x0):
        #hess_LL = lambda y: sum([-self._hesslogf(x, y[0], y[1]) for x in XX])
        grad_LL = lambda y: numpy.array([-self._dlogf(x, y[0], y[1]) for x in XX]).sum(axis=0)
        LL = lambda y: -self.score(XX, mode=y[0], cv=y[1])
        res = minimize(fun=LL, method='BFGS', x0=x0, jac=grad_LL)['x']
        return res

    def fit(self, XX):
        X = numpy.array(XX).copy()
        shape = 1/(1/X - 1/X.mean()).mean()
        self.mode = self._mode(X.mean(), shape)
        self.cv = X.mean()**2/shape
        return self

    def kde(self, X, cv=None):
        if cv:
            pass
        else:
            cv = self.cv
        return lambda t: numpy.mean([self.pdf(t, x, cv) for x in X])

    def sample(self, n_sample=1):
        if n_sample < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at least one sample." % (n_sample)
            )

        # https://fr.wikipedia.org/wiki/Loi_inverse-gaussienne#Simulation_num%C3%A9rique_de_la_loi_inverse-gaussienne
        y = numpy.random.normal(size=n_sample) ** 2
        X = self.mean + (
                self.mean ** 2 * y - self.mean * numpy.sqrt(4 * self.mean * self.shape * y + self.mean ** 2 * y ** 2)) / (
                        2 * self.shape)
        U = numpy.random.rand(n_sample)
        S = numpy.zeros(n_sample)
        Z = self.mean / (self.mean + X)
        ok = (U <= Z)
        notok = (U > Z)
        S[ok] = X[ok]
        S[notok] = self.mean ** 2 / X[notok]
        return S

    def get_parameters(self):
        return {'mode': self.mode, 'cv': self.cv}

