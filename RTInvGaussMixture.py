"""Inverse Gaussian mixtures with shared, fixed response-time dispersion.

The historical parameter ``cv`` is gamma = variance / mean, not the coefficient
of variation.  At fixed utilization rho, backlog = (1 - rho) * mean.
"""

import numbers

import numpy as np

from .rInvGauss import _positive_values, _scalar_or_array, rInvGauss
from .rInvGaussMixture import _validate_data, rInvGaussMixture


class RTInvGaussMixture(rInvGaussMixture):
    """Fit a response-time mixture with fixed utilization and dispersion.

    Parameters
    ----------
    n_components : int
        Positive number of components.
    cv_init : float or array-like
        Shared positive dispersion gamma. A vector must contain one identical
        value per component. This parameter remains fixed during fitting.
    max_iter : int, default=100
        Maximum number of expectation-maximization iterations.
    tol : float, default=1e-4
        Convergence tolerance used by the base mixture estimator.
    modes_init, backlog_init : array-like, optional
        Positive initial modes or backlogs. If both are supplied, they must
        describe the same means for the supplied gamma and utilization.
    weights_init : array-like, optional
        Nonnegative initial weights summing to one.
    verbose : bool, default=False
        Print fitting progress.
    utilization : float
        Fixed utilization rho, with ``0 <= rho < 1``.
    random_state : int or NumPy random generator, optional
        Random state passed to the base estimator.

    Notes
    -----
    For responsibilities z and fixed gamma, let A = sum(z / x) / sum(z).
    The weighted maximum-likelihood mean is the positive root of
    ``A * mean**2 - mean - gamma = 0``. Backlogs and weights are the only fitted
    parameters; an unconstrained numerical optimizer is unnecessary.
    """

    def __init__(self, n_components, cv_init, max_iter=100, tol=1e-4,
                 modes_init=None, backlog_init=None, weights_init=None,
                 verbose=False, utilization=None, random_state=None):
        if (isinstance(n_components, (bool, np.bool_))
                or not isinstance(n_components, numbers.Integral)
                or n_components < 1):
            raise ValueError("n_components must be a positive integer")
        if (isinstance(utilization, (bool, np.bool_))
                or not isinstance(utilization, numbers.Real)
                or not np.isfinite(utilization)
                or not 0 <= utilization < 1):
            raise ValueError("utilization must be finite and satisfy 0 <= utilization < 1")
        self.utilization = float(utilization)
        if np.asarray(cv_init).dtype.kind == "b":
            raise ValueError("cv_init must be a positive dispersion, not a boolean")
        dispersion = _positive_values(cv_init, "cv_init")
        if dispersion.ndim == 0:
            self._fixed_cv = float(dispersion)
        elif dispersion.ndim == 1 and dispersion.size == n_components:
            if not np.all(dispersion == dispersion[0]):
                raise ValueError("RT components must share one fixed cv_init dispersion")
            self._fixed_cv = float(dispersion[0])
        else:
            raise ValueError("cv_init must be a scalar or one identical value per component")

        if backlog_init is not None:
            backlog = _positive_values(backlog_init, "backlog_init")
            if backlog.ndim != 1 or backlog.size != n_components:
                raise ValueError("backlog_init must contain one value per component")
            backlog_modes = self._mode(backlog)
            if modes_init is not None:
                modes = _positive_values(modes_init, "modes_init")
                if (modes.shape != backlog_modes.shape
                        or not np.allclose(modes, backlog_modes, rtol=1e-10, atol=0)):
                    raise ValueError("modes_init and backlog_init describe different components")
            else:
                modes_init = backlog_modes

        super().__init__(
            n_components=n_components, max_iter=max_iter, tol=tol,
            modes_init=modes_init, cv_init=self._fixed_cv,
            weights_init=weights_init, verbose=verbose, random_state=random_state,
        )
        self.backlog_ = (None if self.modes_ is None
                         else self._backlog(self.modes_, self._fixed_cv))

    def _mode(self, backlog=None):
        """Convert backlog to mode without cancellation from subtraction."""
        if backlog is None:
            backlog = self.backlog_
        backlog = _positive_values(backlog, "backlog")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mean = backlog / (1 - self.utilization)
        return rInvGauss._mode_from_mean_dispersion(mean, self._fixed_cv)

    @property
    def _means(self):
        """Component means implied by the current physical backlogs."""
        if self.backlog_ is None:
            raise ValueError("component backlogs are not initialized")
        return np.asarray(self.backlog_, dtype=float) / (1 - self.utilization)

    def _backlog(self, theta, gamma):
        """Convert mode and dispersion to backlog at fixed utilization."""
        theta = _positive_values(theta, "mode")
        gamma = _positive_values(gamma, "dispersion")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            backlog = (1 - self.utilization) * rInvGauss._mean(theta, gamma)
        _positive_values(backlog, "backlog computed from mode")
        return _scalar_or_array(backlog)

    def dtheta_dbacklog(self, backlog):
        """First mode derivative with respect to backlog, holding gamma fixed."""
        mean = _positive_values(backlog, "backlog") / (1 - self.utilization)
        scale = np.maximum(mean, self._fixed_cv)
        radius = np.hypot(mean / scale, 1.5 * (self._fixed_cv / scale))
        return _scalar_or_array((mean / scale / radius) / (1 - self.utilization))

    def d2theta_dbacklog2(self, backlog):
        """Second mode derivative with respect to backlog."""
        mean = _positive_values(backlog, "backlog") / (1 - self.utilization)
        scale = np.maximum(mean, self._fixed_cv)
        radius = np.hypot(mean / scale, 1.5 * (self._fixed_cv / scale))
        derivative = ((1.5 * (self._fixed_cv / scale) / radius) ** 2 / radius / scale
                      / (1 - self.utilization) ** 2)
        return _scalar_or_array(derivative)

    def _update_params(self, XX, zz, x0=None, method=None):
        """Return the exact weighted maximum-likelihood backlog.

        ``x0`` and ``method`` are accepted for compatibility with the former
        optimizer-based implementation; the exact optimum needs neither.
        """
        self._validate_method(method)
        values = _validate_data(XX)
        responsibility = np.asarray(zz, dtype=float)
        if (responsibility.shape != values.shape
                or np.any(~np.isfinite(responsibility))
                or np.any(responsibility < 0)):
            raise ValueError("responsibilities must be finite, nonnegative and match the data")
        active = responsibility > 0
        if not np.any(active):
            raise ValueError("at least one responsibility must be positive")
        values = values[active]
        responsibility = responsibility[active]
        # Scale before summation and reciprocation to handle a wide data range.
        weights = responsibility / responsibility.max()
        weights /= weights.sum()
        smallest = values.min()
        harmonic_mean = smallest / np.dot(weights, smallest / values)
        root_harmonic = np.sqrt(harmonic_mean)
        mean = (harmonic_mean / 2
                + root_harmonic * np.hypot(root_harmonic / 2, np.sqrt(self._fixed_cv)))
        backlog = (1 - self.utilization) * mean
        _positive_values(backlog, "fitted backlog")
        return float(backlog)

    def _validate_initial_dispersion(self):
        """The shared fixed dispersion is not subject to the fitted-gamma floor."""
        return None

    def initialize(self, X, method=None):
        """Initialize the base mixture and synchronize its physical backlogs."""
        super().initialize(X, method=method)
        self.cv_ = np.full(self._n_components, self._fixed_cv)
        self.backlog_ = self._backlog(self.modes_, self._fixed_cv)
        return self

    def _M_step(self, X, z, method=None):
        """Update weights and backlogs while preserving the shared dispersion."""
        masses = np.sum(z, axis=0)
        if not np.isfinite(masses).all() or masses.sum() <= 0:
            raise ValueError("component responsibilities must have positive finite total mass")
        self.weights_ = masses / masses.sum()
        for component, mass in enumerate(masses):
            # An inactive component contributes nothing to the likelihood.
            if mass > 0:
                self.backlog_[component] = self._update_params(
                    X, z[:, component], self.backlog_[component], method=method,
                )
                self.modes_[component] = self._mode(self.backlog_[component])
        return 0

    def _n_parameters(self):
        """Count K backlogs and K-1 weights; utilization and gamma are fixed."""
        return 2 * self._n_components - 1

    def get_parameters(self):
        """Return a serializable snapshot, including the physical backlogs."""
        self._check_parameters()
        return {
            "weights": self.weights_.tolist(), "modes": self.modes_.tolist(),
            "backlog": np.asarray(self.backlog_).tolist(),
            "utilization": self.utilization, "cv": self._fixed_cv,
            "n_components": self._n_components,
        }

    def set_parameters(self, params):
        """Restore a validated snapshot without changing the component count."""
        if not isinstance(params, dict):
            raise ValueError("params must be a parameter dictionary")
        if params.get("n_components", self._n_components) != self._n_components:
            raise ValueError("n_components cannot be changed by set_parameters")
        if not {"weights", "cv"} <= params.keys():
            raise ValueError("params must contain weights, cv and modes or backlog")
        if params.get("modes") is None and params.get("backlog") is None:
            raise ValueError("params must contain modes or backlog")
        # Validate the entire candidate before mutating this estimator.
        candidate = type(self)(
            self._n_components, cv_init=params["cv"],
            max_iter=self.max_iter, tol=self.tol,
            modes_init=params.get("modes"), backlog_init=params.get("backlog"),
            weights_init=params["weights"], verbose=self.verbose,
            utilization=params.get("utilization", self.utilization),
            random_state=self.random_state,
        )
        candidate._check_parameters()
        super().set_parameters({
            "weights": candidate.weights_, "modes": candidate.modes_,
            "cv": candidate.cv_, "n_components": self._n_components,
        })
        self.utilization = candidate.utilization
        self._fixed_cv = candidate._fixed_cv
        self.backlog_ = np.asarray(candidate.backlog_).copy()
        return self
