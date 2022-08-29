import numpy as np
from math import sqrt, log, exp
from scipy.optimize import minimize, fmin_bfgs
from tqdm import trange
from rInvGauss import rInvGauss
from rInvGaussMixture import rInvGaussMixtureCore, rInvGaussMixture
from sklearn.cluster import KMeans


class RTInvGaussMixture(rInvGaussMixtureCore):

    def __init__(self, n_components, smooth_init, max_iter=100, tol=1e-4, modes_init=None,
                 weights_init=None, verbose=False, utilization=None):

        super().__init__(n_components=n_components, tol=tol, max_iter=max_iter, modes_init=modes_init,
                         smooth_init=smooth_init, weights_init=weights_init, verbose=verbose)
        self.utilization = utilization

        if smooth_init is not None and (isinstance(smooth_init, int) or isinstance(smooth_init, float)):
            self.smooth_ = [smooth_init] * n_components

        if smooth_init and modes_init:
            self.backlog_ = [self._backlog(m, smooth_init) for m in modes_init]
        else:
            self.backlog_ = None

    def _mode(self, backlog=None):
        return sqrt((backlog / (1 - self.utilization))**2 + (1.5 * self.smooth_[0])**2) - 1.5 * self.smooth_[0]

    def _backlog(self, theta, gamma):
        return (1 - self.utilization) * sqrt(theta) * sqrt(theta + 3 * gamma)
    
    def dtheta_dbacklog(self, backlog):
        return backlog / (1 - self.utilization) ** 2 / sqrt((backlog / (1 - self.utilization)) ** 2 + (1.5 * self.smooth_[0]) ** 2)
        
    def _update_params(self, XX, zz, x0, method='dogleg'):        
        grad_LL = lambda y: -self.dtheta_dbacklog(y) * self._derivative_complete_likelihood(XX, zz, self._mode(y), self.smooth_[0])[0]
        LL = lambda y: -self._complete_likelihood(XX, zz, self._mode(y), self.smooth_[0])
        res = minimize(fun=LL, method=method, x0=x0, jac=grad_LL)
        return res['x']

    def initialize(self, X, method='kmeans'):
        if method == 'kmeans':
            kmeans = KMeans(self._n_components).fit(X.reshape(-1, 1))

        z = np.zeros((len(X), self._n_components))
        if self.weights_ is None:
            for i, j in enumerate(kmeans.predict(X.reshape(-1, 1))):
                z[i, j] = 1
            self.weights_ = np.mean(z, axis=0).tolist()

        self.modes_ = kmeans.cluster_centers_.reshape(-1)
        self.backlog_ = [self._backlog(m, self.smooth_[0]) for m in self.modes_]

        if self.smooth_ is None:
            self.smooth_ = [1.]

    def _M_step(self, X, z, method='dogleg'):
        # M-step
        self.weights_ = np.mean(z, axis=0).tolist()
        for j in range(self._n_components):
            self.backlog_[j] = self._update_params(X, z[:, j], self.backlog_[j], method=method)
            self.modes_[j] = self._mode(self.backlog_[j])
        return 0

    def get_parameters(self):
        return {'weights': self.weights_, 'modes': self.modes_, 'backlog': self.backlog_,
                'smooth': self.smooth_[0], 'n_components': self._n_components}


if __name__ == '__main__':
    import pandas as pd
    from scipy.stats import chi2
    import matplotlib.pyplot as plt

    C = pd.read_csv('data/execution_times.csv', sep=';')
    C = np.c_[C[["0"]], C[["1"]]]
    execution_times = []
    for x, y in C:
        x = [int(xx) for xx in x[1:-1].split(',')]
        y = [float(yy) for yy in y[1:-1].split(',')]
        execution_times.append((x, y))
    periods = pd.read_csv('data/periods.csv')[["0"]].values.reshape(1, -1)[0]
    rates = [1 / p for p in periods]
    n = len(periods)
    max_c = 0
    for c in execution_times:
        max_tmp = max(c[0])
        max_c = max((max_c, max_tmp))

    # plt.clf()
    # fig, ax = plt.subplots(10, 4, figsize=(20, 40))
    # for (i, c), ax_ in zip(enumerate(execution_times), ax.reshape(-1)):
    #    ax_.bar(*c, color='black')
    #    ax_.set_xlim(1, max_c)
    #    ax_.set_title(r'$\tau_{}, T_{}={}$'.format(str('{')+str(i)+str('}'), str('{')+str(i)+str('}'), periods[i]))
    # plt.savefig('simulations/execution_times.pdf'.format(i))
    # plt.show()

    m = np.array([sum([c0 * c1 for c0, c1 in zip(*c)]) for c in execution_times])
    U = np.array([m[i] / periods[i] for i, _ in enumerate(execution_times)])
    Ubar = np.cumsum(U)
    V = np.cumsum([(sum([c0 ** 2 * c1 for c0, c1 in zip(*c)]) - m[i] ** 2)
                   for i, c in enumerate(execution_times)])
    gamma = V / (1 - Ubar) ** 2

    n_components = 2
    n_task = 30
    sample = pd.read_csv('data/task_{}.csv'.format(n_task))["0"].astype(float)
    rIG = RTInvGaussMixture(n_components=n_components, smooth_init=gamma[n_task],
                            utilization=U[n_task]).fit(sample, method='BFGS')
    print(rIG.get_parameters())
    plt.hist(sample, density=True, bins=50, color='black')
    t_range = np.linspace(0.1, max(sample))
    plt.plot(t_range, rIG.pdf(t_range), color='red', label='gamma fixed')

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

