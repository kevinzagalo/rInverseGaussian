"""Inverse Gaussian distributions parameterized by mode and dispersion.

The historical argument ``cv`` denotes Punzo's dispersion parameter gamma,
not the coefficient of variation. For mean mu and shape lambda,
gamma = mu**2 / lambda and Var[X] = mu * gamma.
"""

import numbers
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import invgauss


def _positive_values(value, name):
    """Convert finite, strictly positive numeric values to a float array."""
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain positive finite numbers") from exc
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must contain positive finite numbers")
    return result


def _positive_scalar(value, name):
    result = _positive_values(value, name)
    if result.ndim != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(result)


def _scalar_or_array(value):
    """Preserve scalar inputs without discarding array dimensions."""
    result = np.asarray(value)
    return float(result) if result.ndim == 0 else result


def _sample_data(values):
    """Accept a univariate sample, including an (n, 1) column vector."""
    data = _positive_values(values, "sample")
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1 or data.size == 0:
        raise ValueError("sample must be a nonempty one-dimensional sequence")
    return data


def _random_generator(random_state):
    """Retain global NumPy seeding when no explicit random state is supplied."""
    if random_state is None:
        return np.random
    if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        return random_state
    if isinstance(random_state, numbers.Integral) and not isinstance(
        random_state, (bool, np.bool_)
    ):
        return np.random.default_rng(random_state)
    raise ValueError("random_state must be None, an integer, or a NumPy RNG")


class rInvGauss:
    """A reparameterized inverse Gaussian distribution.

    Parameters
    ----------
    mode : float, optional
        Positive mode theta. Omit it when estimating parameters with ``fit``.
    cv : float, optional
        Positive dispersion gamma = mu**2 / lambda. Despite its historical
        name, this is not a coefficient of variation.
    tol, max_iter, verbose
        Retained for API compatibility. Fitting uses the exact maximum
        likelihood solution and does not require an iterative optimizer.

    Notes
    -----
    The usual inverse Gaussian parameters are
    mu = sqrt(theta * (theta + 3 * gamma)) and lambda = mu**2 / gamma.
    SciPy uses ``invgauss(mu=mu / lambda, scale=lambda)``.
    """

    def __init__(self, mode=None, cv=None, tol=1e-4, max_iter=100, verbose=False):
        self.mode = None if mode is None else _positive_scalar(mode, "mode")
        self.cv = None if cv is None else _positive_scalar(cv, "cv")
        self.tol = _positive_scalar(tol, "tol")
        if not isinstance(max_iter, numbers.Integral) or isinstance(
            max_iter, (bool, np.bool_)
        ) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.max_iter = int(max_iter)
        self.n_iter = self.max_iter  # Historical name for the iteration limit.
        self.n_iter_ = 0
        self.verbose = bool(verbose)
        self.converged_ = False
        self.fitted_ = False

    @staticmethod
    def _mean(mode, cv):
        theta = _positive_values(mode, "mode")
        gamma = _positive_values(cv, "cv")
        # hypot avoids squaring a large mode before taking its square root.
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.hypot(theta, np.sqrt(theta) * np.sqrt(gamma) * np.sqrt(3.0))
        return _scalar_or_array(_positive_values(mean, "derived mean"))

    @classmethod
    def _shape(cls, mode, cv):
        mean = cls._mean(mode, cv)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            shape = np.square(mean / np.sqrt(_positive_values(cv, "cv")))
        return _scalar_or_array(_positive_values(shape, "derived shape"))

    @staticmethod
    def _mode(mean, shape):
        mean, shape = np.broadcast_arrays(
            _positive_values(mean, "mean"), _positive_values(shape, "shape")
        )
        result = np.empty_like(mean)
        with np.errstate(over="ignore", under="ignore"):
            ratio = mean / shape
        small = ratio <= 2.0 / 3.0
        t = 1.5 * ratio[small]
        # Rationalizing sqrt(1 + t**2) - t removes catastrophic cancellation.
        result[small] = mean[small] / (np.hypot(1.0, t) + t)
        reciprocal = shape[~small] / mean[~small] / 1.5
        result[~small] = shape[~small] / (
            1.5 * (np.hypot(1.0, reciprocal) + 1.0)
        )
        return _scalar_or_array(_positive_values(result, "derived mode"))

    @staticmethod
    def _mode_from_mean_dispersion(mean, cv):
        """Convert (mu, gamma) without overflowing gamma/mu or squaring mu."""
        mean, gamma = np.broadcast_arrays(
            _positive_values(mean, "mean"), _positive_values(cv, "cv")
        )
        result = np.empty_like(mean)
        small = gamma <= mean
        ratio = 1.5 * (gamma[small] / mean[small])
        result[small] = mean[small] / (np.hypot(1.0, ratio) + ratio)
        # When gamma is large, use mu/gamma instead. Multiplying by mu before
        # dividing the final denominator preserves representable tiny modes.
        reciprocal = mean[~small] / gamma[~small]
        result[~small] = (reciprocal * mean[~small]) / (
            np.hypot(reciprocal, 1.5) + 1.5
        )
        return _scalar_or_array(_positive_values(result, "derived mode"))

    @staticmethod
    def _cv(mean, shape):
        mean = _positive_values(mean, "mean")
        shape = _positive_values(shape, "shape")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gamma = np.square(mean / np.sqrt(shape))
        return _scalar_or_array(_positive_values(gamma, "derived cv"))

    def _parameters(self, mode=None, cv=None):
        theta = self.mode if mode is None else mode
        gamma = self.cv if cv is None else cv
        if theta is None or gamma is None:
            raise ValueError("set both mode and cv, or fit the distribution first")
        return _positive_scalar(theta, "mode"), _positive_scalar(gamma, "cv")

    def _scipy_parameters(self, mode=None, cv=None):
        theta, gamma = self._parameters(mode, cv)
        mean = self._mean(theta, gamma)
        shape = self._shape(theta, gamma)
        return {"mu": _positive_scalar(gamma / mean, "SciPy shape"), "scale": shape}

    @property
    def mean(self):
        """Distribution mean mu."""
        return self._mean(*self._parameters())

    @property
    def shape(self):
        """Shape lambda in the conventional inverse Gaussian parameterization."""
        return self._shape(*self._parameters())

    @property
    def variance(self):
        """Distribution variance mu * gamma."""
        _, gamma = self._parameters()
        return self.mean * gamma

    def pdf(self, x, mode=None, cv=None):
        """Evaluate the density; nonpositive inputs have density zero."""
        return invgauss.pdf(x, **self._scipy_parameters(mode, cv))

    def log_pdf(self, x, mode=None, cv=None):
        """Evaluate the log density directly, avoiding density underflow."""
        return invgauss.logpdf(x, **self._scipy_parameters(mode, cv))

    def cdf(self, x):
        """Return P(X <= x), preserving scalar or array input shape."""
        return invgauss.cdf(x, **self._scipy_parameters())

    def logcdf(self, x):
        """Return log P(X <= x), including probabilities that underflow."""
        return invgauss.logcdf(x, **self._scipy_parameters())

    def sf(self, x):
        """Return P(X > x) directly, without subtracting a rounded CDF from 1."""
        return invgauss.sf(x, **self._scipy_parameters())

    def logsf(self, x):
        """Return log P(X > x) for tail calculations."""
        return invgauss.logsf(x, **self._scipy_parameters())

    def quantile(self, alpha):
        """Inverse CDF for probabilities in [0, 1]; NaNs propagate.

        The endpoints are 0 and positive infinity, respectively. SciPy's
        initial result is checked in log probability space; a bracketed solve
        repairs failed or inaccurate extreme-tail inverse evaluations.
        """
        probability = np.asarray(alpha, dtype=float)
        if np.any((probability < 0) | (probability > 1) | np.isinf(probability)):
            raise ValueError("alpha must lie in [0, 1]")
        parameters = self._scipy_parameters()
        result = np.full_like(probability, np.nan)
        result[probability == 0], result[probability == 1] = 0.0, np.inf
        interior = (probability > 0) & (probability < 1)
        probabilities = probability[interior]
        if probabilities.size == 0:
            return _scalar_or_array(result)
        with warnings.catch_warnings(record=True) as reported:
            # Record rather than raise: some Boost ufunc versions mishandle
            # warning-as-error and leave an exception set in the C extension.
            warnings.filterwarnings("always", category=RuntimeWarning,
                                    module=r"scipy\.stats\..*")
            quantiles = invgauss.ppf(probabilities, **parameters)
        inverse_warning = False
        for warning in reported:
            if (issubclass(warning.category, RuntimeWarning)
                    and "/scipy/stats/" in warning.filename.replace("\\", "/")):
                inverse_warning = True
            else:
                warnings.warn_explicit(warning.message, warning.category,
                                       warning.filename, warning.lineno)
        if inverse_warning:
            quantiles = np.full(probabilities.shape, np.nan)

        lower_tail = probabilities <= 0.5
        target = np.where(lower_tail, np.log(probabilities), np.log1p(-probabilities))
        log_probability = np.full(probabilities.shape, np.nan)
        valid = np.isfinite(quantiles) & (quantiles > 0)
        low, high = valid & lower_tail, valid & ~lower_tail
        log_probability[low] = invgauss.logcdf(quantiles[low], **parameters)
        log_probability[high] = invgauss.logsf(quantiles[high], **parameters)
        tolerance = 64 * np.finfo(float).eps * (1 + np.abs(target))
        consistent = valid & (np.abs(log_probability - target) <= tolerance)
        for index in np.flatnonzero(~consistent):
            quantiles[index] = self._solve_quantile(probabilities[index], parameters)
        result[interior] = quantiles
        return _scalar_or_array(result)

    def _solve_quantile(self, probability, parameters):
        """Bracket a quantile using logarithmic tail probabilities."""
        lower_tail = probability <= 0.5
        target = np.log(probability) if lower_tail else np.log1p(-probability)
        log_tail = invgauss.logcdf if lower_tail else invgauss.logsf

        def objective(x):
            return float(log_tail(x, **parameters) - target)

        upper = self.mean
        largest = np.finfo(float).max
        while True:
            residual = objective(upper)
            if np.isnan(residual):
                raise FloatingPointError("The quantile exceeds the reliable numerical range.")
            if (lower_tail and residual >= 0) or (not lower_tail and residual <= 0):
                break
            if upper == largest:
                raise FloatingPointError("The requested quantile exceeds the numerical range.")
            upper = upper * 2 if upper < largest / 2 else largest
        quantile = brentq(objective, 0.0, upper, xtol=np.nextafter(0.0, 1.0),
                          rtol=4 * np.finfo(float).eps, maxiter=2048)
        if not np.isfinite(quantile) or quantile <= 0:
            raise FloatingPointError("The requested quantile is outside the numerical range.")
        return quantile

    def _dlogf(self, x, mode=None, cv=None):
        """Gradient of log f with respect to (theta, gamma)."""
        theta, gamma = self._parameters(mode, cv)
        x = _positive_values(x, "x")
        mu = self._mean(theta, gamma)
        dmu_dtheta = (theta + 1.5 * gamma) / mu
        dmu_dgamma = 1.5 * theta / mu
        dlog_dmu = 1.0 / mu + (x - mu) / (gamma * x)
        dlog_dgamma = -0.5 / gamma + (x - mu) ** 2 / (2 * gamma**2 * x)
        return np.array(
            [dlog_dmu * dmu_dtheta, dlog_dmu * dmu_dgamma + dlog_dgamma]
        )

    def _hesslogf(self, x, mode=None, cv=None):
        """Hessian of log f with respect to (theta, gamma).

        Applying the chain rule through (mu, gamma) is easier to verify than
        a fully expanded expression in theta and gamma.
        """
        theta, gamma = self._parameters(mode, cv)
        x = _positive_values(x, "x")
        mu = self._mean(theta, gamma)
        a = (theta + 1.5 * gamma) / mu
        b = 1.5 * theta / mu
        dlog_dmu = 1.0 / mu + (x - mu) / (gamma * x)
        h_mu_mu = -1.0 / mu**2 - 1.0 / (gamma * x)
        h_mu_gamma = -(x - mu) / (gamma**2 * x)
        h_gamma_gamma = 0.5 / gamma**2 - (x - mu) ** 2 / (gamma**3 * x)
        h_theta_theta = h_mu_mu * a**2 - dlog_dmu * 2.25 * gamma**2 / mu**3
        h_theta_gamma = (
            h_mu_mu * a * b + h_mu_gamma * a
            + dlog_dmu * 2.25 * theta * gamma / mu**3
        )
        h_gamma_total = (
            h_mu_mu * b**2 + 2 * h_mu_gamma * b + h_gamma_gamma
            - dlog_dmu * 2.25 * theta**2 / mu**3
        )
        return np.array(
            [[h_theta_theta, h_theta_gamma], [h_theta_gamma, h_gamma_total]]
        )

    def score(self, X, y=None, mode=None, cv=None):
        """Return the total log likelihood of a positive, finite sample."""
        return float(np.sum(self.log_pdf(_sample_data(X), mode=mode, cv=cv)))

    def _update_params(self, XX, x0=None):
        """Compatibility helper returning the exact MLE; x0 is unnecessary."""
        fitted = rInvGauss().fit(XX)
        return np.array([fitted.mode, fitted.cv])

    def fit(self, XX, y=None):
        """Estimate theta and gamma by maximum likelihood and return ``self``.

        Constant samples have no finite positive-dispersion maximizer and
        therefore raise ``ValueError``. No artificial regularization is
        applied to the single-distribution estimator.
        """
        data = _sample_data(XX)
        scale = float(np.max(data))
        mu = float(scale * np.mean(data / scale))
        # E[(X-mu)**2/X] = mu**2 * (E[1/X] - 1/mu), with no cancellation
        # between nearly equal reciprocal moments for concentrated samples.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gamma = float(np.mean(np.square((data - mu) / np.sqrt(data))))
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(
                "sample has no representable finite positive-dispersion MLE; "
                "at least two distinct observations are required"
            )
        mode = self._mode_from_mean_dispersion(mu, gamma)
        self._shape(mode, gamma)  # Reject an unrepresentable conventional shape.
        self.mode, self.cv = mode, gamma
        self.fitted_ = True
        self.converged_ = True
        self.n_iter_ = 1
        return self

    def kde(self, X, cv=None):
        """Return an inverse Gaussian kernel density estimator.

        Every observation is the mode of a kernel with the supplied common
        dispersion. Evaluating a vector returns one density per coordinate.
        """
        data = _sample_data(X)
        gamma = _positive_scalar(self.cv if cv is None else cv, "cv")
        mean = self._mean(data, gamma)
        shape = self._shape(data, gamma)

        def density(x):
            points = np.asarray(x, dtype=float)
            values = invgauss.pdf(points[..., None], mu=gamma / mean, scale=shape)
            return _scalar_or_array(np.mean(values, axis=-1))

        return density

    def sample(self, n_sample=1, random_state=None):
        """Draw positive observations using NumPy's inverse Gaussian sampler.

        ``random_state`` accepts an integer seed or a NumPy Generator or
        RandomState. With None, ``np.random.seed`` retains its usual effect.
        """
        if not isinstance(n_sample, numbers.Integral) or isinstance(
            n_sample, (bool, np.bool_)
        ) or n_sample < 1:
            raise ValueError("n_sample must be a positive integer")
        generator = _random_generator(random_state)
        return generator.wald(self.mean, self.shape, size=int(n_sample))

    def get_parameters(self):
        """Return the mode and legacy dispersion parameter."""
        return {"mode": self.mode, "cv": self.cv}


class RTInvGauss(rInvGauss):
    """Inverse Gaussian response-time model with fixed utilization.

    For backlog b, deviation d, and utilization u, the mean is b / (1-u),
    the shape is b**2 / d**2, and gamma is d**2 / (1-u)**2.
    Utilization must satisfy 0 <= u < 1. To estimate backlog and deviation
    from a sample, initialize with utilization alone and call ``fit``.
    """

    def __init__(
        self, backlog=None, deviation=None, utilization=None,
        tol=1e-4, max_iter=100, verbose=False,
    ):
        try:
            u = np.asarray(utilization, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("utilization must be a finite scalar in [0, 1)") from exc
        if u.ndim != 0 or not np.isfinite(u) or not 0 <= u < 1:
            raise ValueError("utilization must be a finite scalar in [0, 1)")
        self.utilization = float(u)
        if (backlog is None) != (deviation is None):
            raise ValueError("backlog and deviation must be supplied together")
        if backlog is None:
            self.backlog = self.deviation = None
            mode = gamma = None
        else:
            self.backlog = _positive_scalar(backlog, "backlog")
            self.deviation = _positive_scalar(deviation, "deviation")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                mean = np.divide(self.backlog, 1 - self.utilization)
                gamma = np.square(np.divide(self.deviation, 1 - self.utilization))
            mode = self._mode_from_mean_dispersion(mean, gamma)
        super().__init__(mode=mode, cv=gamma, tol=tol, max_iter=max_iter, verbose=verbose)

    def fit(self, XX, y=None):
        """Fit at fixed utilization and update the physical parameters."""
        super().fit(XX, y=y)
        self.backlog = (1 - self.utilization) * self.mean
        self.deviation = (1 - self.utilization) * np.sqrt(self.cv)
        return self

    def get_parameters(self):
        """Return both distribution and response-time parameters."""
        return {
            **super().get_parameters(),
            "backlog": self.backlog,
            "deviation": self.deviation,
            "utilization": self.utilization,
        }
