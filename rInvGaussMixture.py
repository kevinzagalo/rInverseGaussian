"""Mixtures of inverse Gaussian distributions parameterized by mode and dispersion.

The legacy name "cv" means gamma = variance / mean. Weighted maximum-likelihood
updates are computed in the equivalent (mean, dispersion) parameterization.
"""

from numbers import Integral, Real
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, xlogy
from scipy.stats import kstest
from sklearn.cluster import KMeans

from .rInvGauss import _random_generator, rInvGauss


def _validate_data(X, min_samples=1):
    """Return a copy of positive, finite, univariate observations."""
    data = np.asarray(X, dtype=float)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1 or data.size < min_samples:
        raise ValueError(f"X must contain at least {min_samples} univariate observations.")
    if not np.all(np.isfinite(data)) or np.any(data <= 0):
        raise ValueError("X must contain only finite, strictly positive observations.")
    return data.copy()


def _positive_vector(value, size, name, allow_scalar=False):
    if value is None:
        return None
    if np.asarray(value).dtype.kind == "b":
        raise ValueError(f"{name} must contain numbers, not booleans.")
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain positive finite numbers.") from exc
    if allow_scalar and values.ndim == 0:
        values = np.full(size, float(values))
    if values.shape != (size,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must contain {size} finite, strictly positive values.")
    return values.copy()


def _validate_weights(value, size):
    if value is None:
        return None
    weights = np.asarray(value, dtype=float)
    if (weights.shape != (size,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=1e-7, atol=1e-12)):
        raise ValueError(f"weights must contain {size} nonnegative values summing to one.")
    # Normalize harmless input round-off, while rejecting invalid probability vectors.
    return weights.copy() / weights.sum()


def _fixed_dispersion_mean(data, probabilities, dispersion):
    """Positive root of A*mu**2 - mu - gamma = 0, A = weighted mean(1/X)."""
    # Scaling the harmonic mean avoids overflowing 1/X for tiny observations.
    smallest = np.min(data)
    harmonic_mean = smallest / np.dot(probabilities, smallest / data)
    half = 0.5 * harmonic_mean
    return half + np.hypot(half, np.sqrt(dispersion) * np.sqrt(harmonic_mean))


def _mode_from_mean(mean, dispersion):
    """Rationalized inverse transformation avoids subtractive cancellation."""
    return rInvGauss._mode_from_mean_dispersion(mean, dispersion)


class rInvGaussMixture:
    """Finite mixture estimated by expectation-maximization (EM).

    Parameters
    ----------
    n_components : int
        Number of components.
    max_iter : int, default=100
        Maximum EM iterations.
    tol : float, default=1e-4
        Tolerance on the change in mean log likelihood.
    modes_init, cv_init, weights_init : array-like, optional
        Initial modes, dispersions, and probabilities. Scalar cv_init is
        broadcast to every component. Inputs are copied and respected.
    verbose : bool, default=False
        Print the log likelihood after each iteration.
    random_state : int or NumPy random generator, optional
        Controls initialization and sampling; None uses NumPy's global RNG.
    min_cv : float, default=1e-8
        Absolute lower bound on fitted dispersion, in the units of X, preventing
        component collapse. The mean is reoptimized when this bound is active.

    Notes
    -----
    "cv" is gamma = Var(X) / E[X], not the coefficient of variation. EM finds
    local optima; compare several initializations for overlapping components.
    Fully specified models can be evaluated without calling fit.
    """

    def __init__(self, n_components, max_iter=100, tol=1e-4, modes_init=None,
                 cv_init=None, weights_init=None, verbose=False,
                 random_state=None, min_cv=1e-8):
        for name, value in (("n_components", n_components), ("max_iter", max_iter)):
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (not isinstance(tol, Real) or isinstance(tol, (bool, np.bool_))
                or not np.isfinite(tol) or tol < 0):
            raise ValueError("tol must be finite and nonnegative.")
        if (not isinstance(min_cv, Real) or isinstance(min_cv, (bool, np.bool_))
                or not np.isfinite(min_cv) or min_cv <= 0):
            raise ValueError("min_cv must be finite and strictly positive.")
        _random_generator(random_state)
        self._n_components = int(n_components)
        self.max_iter, self.tol, self.min_cv = int(max_iter), float(tol), float(min_cv)
        self.verbose, self.random_state = verbose, random_state
        self.modes_init = _positive_vector(modes_init, n_components, "modes_init")
        self.cv_init = _positive_vector(cv_init, n_components, "cv_init", allow_scalar=True)
        self.weights_init = _validate_weights(weights_init, n_components)
        self.modes_ = None if self.modes_init is None else self.modes_init.copy()
        self.cv_ = None if self.cv_init is None else self.cv_init.copy()
        self.weights_ = None if self.weights_init is None else self.weights_init.copy()
        self._reset_fit_state()

    @property
    def n_components(self):
        """Number of mixture components."""
        return self._n_components

    def _reset_fit_state(self):
        self.converged_, self.fitted_, self.n_iter_ = False, False, 0
        self.log_likelihood_history_ = []
        self.lower_bound_ = -np.inf

    def _check_parameters(self):
        if self.modes_ is None or self.cv_ is None or self.weights_ is None:
            raise ValueError("Fit the model or supply modes, dispersions, and weights first.")
        self.modes_ = _positive_vector(self.modes_, self._n_components, "modes")
        self.cv_ = _positive_vector(self.cv_, self._n_components, "cv", allow_scalar=True)
        self.weights_ = _validate_weights(self.weights_, self._n_components)

    def _validate_initial_dispersion(self):
        """The initial model must satisfy the same constraint as every M-step."""
        if self.cv_init is not None and np.any(self.cv_init < self.min_cv):
            raise ValueError("cv_init must be >= min_cv when fitting; lower min_cv or change cv_init.")

    @staticmethod
    def _validate_method(method):
        if method in (None, "closed-form"):
            return
        if method not in ("dogleg", "BFGS", "minimize_scalar", "newton"):
            raise ValueError("method must be None or 'closed-form'; legacy optimizer names are deprecated.")
        warnings.warn("The method argument is deprecated; exact M-step updates are used.",
                      DeprecationWarning, stacklevel=3)

    def _component_values(self, x, method):
        self._check_parameters()
        # Vectorize observations, looping only over the usually small component axis.
        return np.stack([getattr(rInvGauss(m, c), method)(x)
                         for m, c in zip(self.modes_, self.cv_)], axis=-1)

    def _log_weights(self):
        result = np.full(self._n_components, -np.inf)
        np.log(self.weights_, out=result, where=self.weights_ > 0)
        return result

    def _validate_component(self, component):
        if (not isinstance(component, Integral) or isinstance(component, (bool, np.bool_))
                or not 0 <= component < self._n_components):
            raise ValueError("component must be a valid zero-based component index.")

    def _mean(self, k):
        self._check_parameters()
        self._validate_component(k)
        return rInvGauss(self.modes_[k], self.cv_[k]).mean

    def _proba_components(self, x):
        """Weighted component densities, distinct from posterior probabilities."""
        values = self._component_values(x, "log_pdf")
        return np.exp(values + self._log_weights())

    def log_pdf(self, x):
        """Log density, using log-sum-exp to avoid density underflow."""
        values = self._component_values(x, "log_pdf")
        return logsumexp(values + self._log_weights(), axis=-1)

    def pdf(self, x):
        """Mixture density for scalar or array-like x."""
        return np.exp(self.log_pdf(x))

    def cdf(self, x):
        """Cumulative probability, summing all weighted component CDFs."""
        values = self._component_values(x, "cdf")
        return np.clip(np.sum(values * self.weights_, axis=-1), 0.0, 1.0)

    def logsf(self, x):
        """Log survival probability, retaining precision in the right tail."""
        values = self._component_values(x, "logsf")
        return np.minimum(logsumexp(values + self._log_weights(), axis=-1), 0.0)

    def sf(self, x):
        """P(X > x), evaluated without subtracting from one."""
        return np.exp(self.logsf(x))

    def dmp(self, deadline):
        """Deadline-miss probability P(X > deadline) under this mixture."""
        return self.sf(deadline)

    def quantile(self, alpha, component=None):
        """Mixture quantiles, or component quantiles when an index is supplied."""
        self._check_parameters()
        if component is not None:
            self._validate_component(component)
            return rInvGauss(self.modes_[component], self.cv_[component]).quantile(alpha)
        probabilities = np.asarray(alpha, dtype=float)
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("alpha must contain finite probabilities in [0, 1].")
        active = [rInvGauss(m, c) for m, c, w in zip(self.modes_, self.cv_, self.weights_) if w > 0]

        def solve(probability):
            if probability == 0:
                return 0.0
            if probability == 1:
                return np.inf
            bounds = [float(model.quantile(probability)) for model in active]
            lower, upper = min(bounds), max(bounds)
            if lower == upper:
                return lower
            # Survival probabilities retain precision when alpha approaches one.
            objective = ((lambda x: self.sf(x) - (1.0 - probability)) if probability > 0.5
                         else (lambda x: self.cdf(x) - probability))
            left, right = objective(lower), objective(upper)
            if left != 0 and right != 0 and np.signbit(left) == np.signbit(right):
                lower, upper = 0.0, upper * 2.0
            if not np.isfinite(upper):
                raise FloatingPointError("The requested quantile exceeds the numerical range.")
            return brentq(objective, lower, upper, xtol=np.nextafter(0.0, 1.0),
                          rtol=4 * np.finfo(float).eps, maxiter=200)

        result = np.array([solve(p) for p in probabilities.reshape(-1)]).reshape(probabilities.shape)
        return float(result) if result.ndim == 0 else result

    def normalize(self, x, y):
        """Compute (x - mean_y)^2 / (gamma_y*x).

        Squaring loses the side of the mean, so this transform alone cannot
        determine a one-sided deadline-miss probability.
        """
        self._validate_component(y)
        values = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("x must be finite and strictly positive.")
        return ((values - self._mean(y)) / np.sqrt(values) / np.sqrt(self.cv_[y])) ** 2

    def score_samples(self, X):
        """Log density of each positive observation."""
        return self.log_pdf(_validate_data(X))

    def score(self, X, y=None):
        """Total log likelihood, not the per-observation average."""
        return float(np.sum(self.score_samples(X)))

    def likelihood(self, X):
        """Likelihood; prefer score because the likelihood can underflow."""
        return float(np.exp(self.score(X)))

    def predict_proba(self, X):
        """Posterior probabilities with shape (n_samples, n_components)."""
        data = _validate_data(X)
        log_density = self._component_values(data, "log_pdf")
        largest = np.max(log_density[:, self.weights_ > 0], axis=1, keepdims=True)
        if not np.all(np.isfinite(largest)):
            raise FloatingPointError("Component log densities exceed the numerical range for X.")
        # Shift before adding log weights: a huge negative log density must not
        # erase the O(1) mixing weights or the normalization constant.
        log_joint = log_density - largest + self._log_weights()
        normalizer = logsumexp(log_joint, axis=1, keepdims=True)
        if not np.all(np.isfinite(normalizer)):
            raise FloatingPointError("Component log densities exceed the numerical range for X.")
        return np.exp(log_joint - normalizer)

    def predict(self, X):
        """Most probable component for each observation."""
        return self.predict_proba(X).argmax(axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).predict(X)

    def _update_weights(self, X):
        """Legacy name for the E-step responsibilities."""
        return self.predict_proba(X)

    @staticmethod
    def _observation_weights(weights, size):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (size,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Observation weights must be finite, nonnegative, and match X.")
        return weights

    def _complete_likelihood(self, X, zz, mode, cv):
        data = _validate_data(X)
        weights = self._observation_weights(zz, data.size)
        active = weights > 0
        return float(np.dot(weights[active], rInvGauss(mode, cv).log_pdf(data[active])))

    def _derivative_complete_likelihood(self, X, zz, mode, cv):
        data = _validate_data(X)
        weights = self._observation_weights(zz, data.size)
        active = weights > 0
        return np.sum(rInvGauss(mode, cv)._dlogf(data[active]) * weights[active], axis=-1)

    def _second_derivative_complete_likelihood(self, X, zz, mode, cv):
        data = _validate_data(X)
        weights = self._observation_weights(zz, data.size)
        active = weights > 0
        return np.sum(rInvGauss(mode, cv)._hesslogf(data[active]) * weights[active], axis=-1)

    def _score_complete(self, X, z):
        data = _validate_data(X)
        responsibilities = np.asarray(z, dtype=float)
        if (responsibilities.shape != (data.size, self._n_components)
                or not np.all(np.isfinite(responsibilities)) or np.any(responsibilities < 0)):
            raise ValueError("z must have nonnegative responsibilities of shape (n_samples, n_components).")
        log_density = self._component_values(data, "log_pdf")
        terms = np.zeros_like(log_density)
        np.multiply(responsibilities, log_density, out=terms, where=responsibilities > 0)
        return float(np.sum(terms + xlogy(responsibilities, self.weights_)))

    def _update_params(self, XX, zz, x0=None, method=None):
        """Exact weighted MLE, with a constrained update at the dispersion floor."""
        self._validate_method(method)
        data = _validate_data(XX)
        weights = self._observation_weights(zz, data.size)
        active = weights > 0
        if not np.any(active):
            if x0 is None:
                raise ValueError("At least one observation weight must be positive.")
            return np.asarray(x0, dtype=float).copy()
        data, weights = data[active], weights[active]
        probabilities = weights / weights.max()
        probabilities /= probabilities.sum()
        scale = data.max()
        mean = scale * np.dot(probabilities, data / scale)
        # Nonnegative residuals avoid cancellation in mu**2*E[1/X] - mu.
        dispersion = np.dot(probabilities, ((data - mean) / np.sqrt(data)) ** 2)
        if dispersion < self.min_cv:
            dispersion = self.min_cv
            # Clipping gamma alone would not maximize the constrained likelihood.
            mean = _fixed_dispersion_mean(data, probabilities, dispersion)
        mode = _mode_from_mean(mean, dispersion)
        if not np.isfinite(mode) or mode <= 0 or not np.isfinite(dispersion):
            raise FloatingPointError("Fitted parameters exceed the numerical range; rescale X.")
        return np.array([mode, dispersion])

    def initialize(self, X, method=None):
        """Initialize missing parameters from one-dimensional k-means clusters."""
        if method not in (None, "kmeans"):
            raise ValueError("The supported initialization method is 'kmeans'.")
        data = _validate_data(X, min_samples=self._n_components)
        self._reset_fit_state()
        self._validate_initial_dispersion()
        if self.modes_init is None:
            if np.unique(data).size < self._n_components:
                raise ValueError("K-means initialization requires n_components distinct observations.")
            if self._n_components == 1:
                labels = np.zeros(data.size, dtype=int)
            else:
                rng = _random_generator(self.random_state)
                seed = int(rng.integers(0, 2**31 - 1)) if isinstance(rng, np.random.Generator) else rng
                seed = None if rng is np.random else seed
                labels = KMeans(self._n_components, n_init=10, random_state=seed).fit_predict(data[:, None])
        else:
            labels = np.abs(data[:, None] - self.modes_init).argmin(axis=1)
        modes, dispersions = [], []
        for component in range(self._n_components):
            subset = data[labels == component]
            if subset.size == 0:
                subset = data
            if self.cv_init is None:
                mode, dispersion = self._update_params(subset, np.ones(subset.size), method=None)
            else:
                dispersion = self.cv_init[component]
                mean = _fixed_dispersion_mean(subset, np.full(subset.size, 1.0 / subset.size), dispersion)
                mode = _mode_from_mean(mean, dispersion)
            modes.append(mode if self.modes_init is None else self.modes_init[component])
            dispersions.append(dispersion)
        self.modes_, self.cv_ = np.asarray(modes), np.asarray(dispersions)
        counts = np.bincount(labels, minlength=self._n_components).astype(float)
        # Empty user-specified components must still have a chance during EM.
        counts = np.maximum(counts, 1.0)
        self.weights_ = counts / counts.sum() if self.weights_init is None else self.weights_init.copy()
        self._check_parameters()
        return self

    def _M_step(self, X, z, method=None):
        self.weights_ = np.mean(z, axis=0)
        for component in range(self._n_components):
            if self.weights_[component] > 0:
                self.modes_[component], self.cv_[component] = self._update_params(
                    X, z[:, component], (self.modes_[component], self.cv_[component]), method=method)
        return 0

    def _EM(self, X, verbose=False, method=None):
        previous = self.score(X)
        if not np.isfinite(previous):
            raise FloatingPointError("Initial log likelihood is not finite; check scaling and initialization.")
        self.log_likelihood_history_ = [previous]
        for iteration in range(1, self.max_iter + 1):
            responsibilities = self._update_weights(X)
            self._M_step(X, responsibilities, method=method)
            current = self.score(X)
            improvement = current - previous
            if not np.isfinite(current) or improvement < -1e-10 * (1 + abs(previous)):
                raise FloatingPointError("EM failed to increase a finite likelihood; rescale X or change initialization.")
            self.n_iter_ = iteration
            self.log_likelihood_history_.append(current)
            self.lower_bound_ = current / len(X)
            if verbose or self.verbose:
                print(f"EM iteration {iteration}: log likelihood = {current:.10g}")
            if abs(improvement) / len(X) <= self.tol:
                self.converged_ = True
                break
            previous = current
        self.fitted_ = True
        if not self.converged_:
            warnings.warn("EM reached max_iter before convergence; inspect converged_ and the likelihood history.",
                          RuntimeWarning, stacklevel=3)
        return self

    def fit(self, X, y=None, verbose=False, method=None):
        """Fit positive data by EM; y is accepted for estimator-style compatibility."""
        self._reset_fit_state()
        self._validate_method(method)
        data = _validate_data(X, min_samples=self._n_components)
        self.initialize(data)
        return self._EM(data, verbose=verbose)

    def _n_parameters(self):
        return 3 * self._n_components - 1

    def aic(self, X):
        """Akaike criterion: 2*p - 2*log_likelihood (lower is better)."""
        return 2 * self._n_parameters() - 2 * self.score(X)

    def bic(self, X):
        """Bayesian criterion: p*log(n) - 2*log_likelihood."""
        data = _validate_data(X)
        return self._n_parameters() * np.log(data.size) - 2 * self.score(data)

    def sample(self, n_sample=1, random_state=None):
        """Draw positive samples; accept a seed or generator for reproducibility."""
        self._check_parameters()
        if not isinstance(n_sample, Integral) or isinstance(n_sample, (bool, np.bool_)) or n_sample < 1:
            raise ValueError("n_sample must be a positive integer.")
        rng = _random_generator(self.random_state if random_state is None else random_state)
        labels = rng.choice(self._n_components, size=n_sample, p=self.weights_)
        means = np.array([rInvGauss(m, c).mean for m, c in zip(self.modes_, self.cv_)])
        shapes = rInvGauss._shape(self.modes_, self.cv_)
        return rng.wald(means[labels], shapes[labels])

    def ks_test(self, sample):
        """Return a whole-mixture KS diagnostic without refitting.

        The nominal p-value assumes a fully specified distribution. It is not
        calibrated for parameters fitted to this sample; use a parametric
        bootstrap or independent validation data for inference.
        """
        data = _validate_data(sample)
        self._check_parameters()
        return kstest(data, self.cdf)

    def get_parameters(self):
        """Return independent, JSON-serializable parameter lists."""
        self._check_parameters()
        return {"weights": self.weights_.tolist(), "modes": self.modes_.tolist(),
                "cv": self.cv_.tolist(), "n_components": self._n_components}

    def set_parameters(self, params):
        """Validate parameters and reuse them for subsequent initialization."""
        if not isinstance(params, dict) or not {"modes", "cv", "weights"} <= params.keys():
            raise ValueError("params must be a dictionary containing modes, cv, and weights.")
        if params.get("n_components", self._n_components) != self._n_components:
            raise ValueError("n_components must match this model.")
        modes = _positive_vector(params["modes"], self._n_components, "modes")
        dispersions = _positive_vector(params["cv"], self._n_components, "cv", allow_scalar=True)
        weights = _validate_weights(params["weights"], self._n_components)
        if modes is None or dispersions is None or weights is None:
            raise ValueError("modes, cv, and weights must all be supplied.")
        self.modes_, self.cv_, self.weights_ = modes, dispersions, weights
        self.modes_init, self.cv_init, self.weights_init = modes.copy(), dispersions.copy(), weights.copy()
        self._reset_fit_state()
        self.fitted_ = True
        return self


