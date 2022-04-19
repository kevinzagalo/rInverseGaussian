import numpy
from math import sqrt, log, exp
from scipy.optimize import minimize, fmin_bfgs
from tqdm import tqdm
from sklearn.mixture._base import BaseMixture


class rInvGaussMixture:

    def __init__(self, n_components=None, max_iter=100, tol=1e-4, modes_init=None, shapes_init=None, weights_init=None):
        self.tol = tol,
        self.n_iter_ = max_iter

        if n_components:
            self._n_components = n_components
        elif modes_init:
            self._n_components = len(modes_init)

        self.modes_ = modes_init
        self.shapes_ = shapes_init
        self.weights_ = weights_init
        self.converged_ = False

    def mu(self, theta, gamma):
        return sqrt(theta * (3 * gamma + theta))

    def lambd(self, theta, gamma):
        return theta * (3 * gamma + theta) / gamma

    def invGauss_pdf(self, x, theta, gamma):
        a1 = sqrt(self.lambd(theta, gamma) / (2 * numpy.pi * x**3))
        a2 = self.lambd(theta, gamma) * (x - self.mu(theta, gamma)) / (x * self.mu(theta, gamma)**2)
        return a1 * exp(-a2/2)

    def log_invGauss_pdf(self, x, theta, gamma):
        a1 = log(self.lambd(theta, gamma))/2 - log(2) - log(numpy.pi) - 3 * log(x)
        a2 = self.lambd(theta, gamma) * (x - self.mu(theta, gamma)) / (x * self.mu(theta, gamma) ** 2)
        return a1 - a2/2

    def _proba_components(self, x):
        return [pi_j * self.invGauss_pdf(x, self.modes_[j], self.shapes_[j]) for j, pi_j in enumerate(self.weights_)]

    def pdf(self, x):
        return sum(self._proba_components(x))

    def _dlogf(self, x, theta, gamma):
        p = 3 * gamma + theta
        dLL_dtheta = - 3/(2*x) \
                     - theta/(x*gamma) \
                     + 1/p + 3*gamma/(2*theta*p) \
                     + sqrt(theta/p)/(2*gamma) \
                     + sqrt(p/theta)/(2*gamma)
        dLL_dgamma = x/(2*gamma**2) \
                     + theta**2/(2*x*gamma**2) \
                     - theta/(2*gamma*p) \
                     + 3*sqrt(theta/p)/(2*gamma) \
                     - sqrt(theta*p)/gamma**2
        return numpy.array([dLL_dtheta, dLL_dgamma])

    def _hesslogf(self, x, theta, gamma):
        p = 3*gamma+theta
        dLL_dtheta2 = -0.25*(4/(x*gamma) + 2/theta**2 + 2/p**2 + 9*gamma/sqrt(theta*p)**3)
        dLL_dgamma2 = -x/gamma**3 - theta**2/(x*gamma**3) \
                      + 3*theta/(2*gamma*p**2) \
                      - 9*sqrt(theta)/(4*gamma*sqrt(p**3))\
                      + theta/(2*gamma**2*p)\
                      - 3*sqrt(theta)/(gamma**2*sqrt(p))\
                      + 2*sqrt(theta*p)/gamma**3
        dLL_dtheta_dgamma = theta/(x*gamma**2) \
                            - (27*gamma**3 + 30*gamma*theta**2\
                                + 4*theta**3\
                                + 3*gamma**2 * (21*theta + 2*sqrt(theta*p)))\
                              / (4*gamma**2*sqrt(theta*p**5))
        return numpy.matrix([[dLL_dtheta2, dLL_dtheta_dgamma], [dLL_dtheta_dgamma, dLL_dgamma2]])

    def _complete_likelihood(self, X, zz, theta, gamma):
        return sum([zz[i] * self.log_invGauss_pdf(x_i, theta, gamma) for i, x_i in enumerate(X)])

    def _derivative_complete_likelihood(self, X, zz, theta, gamma):
        return numpy.array([zz[i] * self._dlogf(x_i, theta, gamma) for i, x_i in enumerate(X)]).sum(axis=0)

    def _second_derivative_complete_likelihood(self, X, zz, theta, gamma):
        return sum([zz[i] * self._hesslogf(x_i, theta, gamma) for i, x_i in enumerate(X)])

    def _update_weights(self, X):
        zz = numpy.zeros((len(X), self._n_components))
        for i, x_i in enumerate(X):
            zz[i, :] = numpy.array(self._proba_components(x_i)) / self.pdf(x_i)
        return zz

    def _update_params(self, X, zz, x0):
        hess_LL = lambda x: -self._second_derivative_complete_likelihood(X, zz, x[0], x[1])
        grad_LL = lambda x: -self._derivative_complete_likelihood(X, zz, x[0], x[1])
        LL = lambda x: -self._complete_likelihood(X, zz, x[0], x[1])
        res = minimize(fun=LL, method='Newton-CG', x0=x0, jac=grad_LL, hess=hess_LL)
        #return fmin_bfgs(f=LL, x0=x0, fprime=grad_LL, disp=False)
        return res['x']

    def _score_complete(self, X, z):
        l1 = sum([sum([z[i, j] * log(pi_j) for j, pi_j in enumerate(self.weights_)]) for i, _ in enumerate(X)])
        l2 = sum([sum([z[i, j] * self.log_invGauss_pdf(x_i, self.modes_[j], self.shapes_[j])
                         for i, x_i in enumerate(X)]) for j in range(self._n_components)])
        return l1 + l2

    def score_sample(self, X):
        return [sum([self.weights_[j] * self.invGauss_pdf(x_i, self.modes_[j], self.shapes_[j])
                     for j in range(self._n_components)]) for i, x_i in enumerate(X)]

    def score(self, X, y=None):
        return sum(self.score_sample(X))

    def _EM(self, X, verbose=False):
        self.weights_ = [1/self._n_components] * self._n_components
        self.modes_ = [50] * self._n_components
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
            mu[i] = self.mu(self.modes_[k], self.shapes_[k])
            lambd[i] = self.lambd(self.modes_[k], self.shapes_[k])
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
    sample = rInvGaussMixture(modes_init=[10, 50], weights_init=[0.3, 0.7], shapes_init=[1, 2]).sample(100)
    model = rInvGaussMixture(2).fit(sample, verbose=True)
    print(model.get_parameters())
    #y = model._second_derivative_complete_likelihood(sample, [1]*len(sample), 1, 1)
    #print(y)

    #rt = numpy.array(RT[0])
    #bic_list = []

    #for k in range(1, 10):
    #    model = rInvGaussMixture(k).fit(rt)
    #    bic_list.append(model.bic(rt))

    #plt.plot(bic_list)
    #plt.show()