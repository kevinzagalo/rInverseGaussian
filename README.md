# rInverseGaussian

Python models for positive observations using the **reparameterized inverse
Gaussian distribution**: a single distribution, finite mixtures, and a
backlog/utilization parameterization for response-time data.

The distribution and kernel estimator follow Antonio Punzo's mode-based
parameterization [1]. The response-time classes retain this repository's
adaptation for fixed-priority real-time systems. Fitting uses closed-form
inverse Gaussian maximum-likelihood updates, including weighted updates in an
expectation-maximization (EM) algorithm.

## Installation

Requires **Python 3.10 or later**, NumPy, SciPy, and scikit-learn.

```bash
git clone https://github.com/kevinzagalo/rInverseGaussian.git
cd rInverseGaussian
python -m pip install .
```

For development, install an editable checkout:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

There is no dependency on SimSo. The four classes can be imported directly:

```python
from rInverseGaussian import (
    rInvGauss,
    rInvGaussMixture,
    RTInvGauss,
    RTInvGaussMixture,
)
```

The original module imports, such as
`from rInverseGaussian.rInvGauss import rInvGauss`, also remain available.

## Parameterization

For $X>0$, the ordinary inverse Gaussian density with mean $\mu>0$ and shape
$\lambda>0$ is

$$
f(x;\mu,\lambda)=\sqrt{\frac{\lambda}{2\pi x^3}}
\exp\!\left[-\frac{\lambda(x-\mu)^2}{2\mu^2x}\right].
$$

The mode-based parameterization [1, Eq. (5)] uses

$$
\theta=\operatorname{mode}(X),\qquad
\gamma=\frac{\mu^2}{\lambda},\qquad
\mu=\sqrt{\theta(\theta+3\gamma)},\qquad
\lambda=\frac{\theta(\theta+3\gamma)}{\gamma}.
$$

**The legacy argument `cv` means the dispersion $\gamma$, not the coefficient
of variation.** Consequently,

$$
\operatorname{Var}(X)=\mu\gamma,\qquad
\operatorname{CV}(X)=\frac{\operatorname{sd}(X)}{\mathbb E[X]}
=\sqrt{\frac{\gamma}{\mu}}.
$$

| API name | Mathematical quantity |
| --- | --- |
| `mode`, `modes_` | Mode $\theta$, or component modes |
| `cv`, `cv_`, `cv_init` | Dispersion $\gamma=\mu^2/\lambda$ |
| `mean` | Mean $\mu$ of a single distribution |
| `shape` | Inverse Gaussian shape $\lambda$ |
| `variance` | Variance $\mu\gamma$ of a single distribution |
| `weights_` | Mixture probabilities $\pi_k$, summing to one |

SciPy uses a different argument convention [2]. The equivalent frozen SciPy
distribution is `scipy.stats.invgauss(mu=mean / shape, scale=shape)`; its `mu`
argument is not the physical mean.

## A single distribution

```python
import numpy as np
from rInverseGaussian import rInvGauss

distribution = rInvGauss(mode=2.0, cv=0.4)
x = np.array([0.5, 1.0, 2.0, 4.0])

print(distribution.pdf(x))
print(distribution.cdf(x))
print(distribution.sf(x))             # P(X > x)
print(distribution.quantile(0.99))    # 99th percentile
print(distribution.mean, distribution.variance)

observations = distribution.sample(2_000, random_state=42)
estimate = rInvGauss().fit(observations)
print(estimate.get_parameters())
print(estimate.score(observations))   # Total log-likelihood
```

`pdf`, `log_pdf`, `cdf`, `logcdf`, `sf`, `logsf`, and `quantile` support scalar
and array inputs. Use `sf` or `logsf` directly for upper-tail calculations: forming
`1 - cdf(x)` can lose precision when the CDF is close to one [2].
Quantiles are checked against their target probabilities in log space. A
bracketed inversion of `logcdf` or `logsf` provides a fallback when the initial
SciPy inverse evaluation fails or is inaccurate in an extreme tail.

An inverse Gaussian kernel density estimate is also available [1, Eq. (9)]:

```python
kernel_density = estimate.kde(observations, cv=0.15)
grid = np.linspace(0.01, 6.0, 100)
density = kernel_density(grid)
```

Each observation becomes the mode of one normalized kernel. The supplied `cv`
is the common smoothing dispersion; this function does not select it by
cross-validation.

### Maximum likelihood for one component

For positive, nonconstant observations, the implemented estimates are

$$
\widehat\mu=\frac1n\sum_i x_i,\qquad
\widehat\gamma=\frac1n\sum_i\frac{(x_i-\widehat\mu)^2}{x_i},\qquad
\widehat\lambda=\frac{\widehat\mu^2}{\widehat\gamma}.
$$

These follow by maximizing the ordinary inverse Gaussian likelihood and
transforming the estimates to $(\theta,\gamma)$. In particular,
$[n^{-1}\sum_i x_i^{-1}-\widehat\mu^{-1}]^{-1}$ estimates $\lambda$, not
$\gamma$. No numerical optimizer is needed. A constant sample has no finite
positive-dispersion maximum-likelihood estimate and is rejected by
`rInvGauss.fit`.

## Finite mixtures

A $K$-component mixture has density
$p(x)=\sum_{k=1}^K\pi_k f(x;\theta_k,\gamma_k)$.

```python
import numpy as np
from rInverseGaussian import rInvGaussMixture

truth = rInvGaussMixture(
    n_components=2,
    modes_init=[1.0, 4.0],
    cv_init=[0.15, 0.35],
    weights_init=[0.65, 0.35],
)
observations = truth.sample(2_000, random_state=42)

model = rInvGaussMixture(
    n_components=2,
    max_iter=300,
    tol=1e-6,
    random_state=42,
).fit(observations)

posterior = model.predict_proba(observations[:5])
assert np.allclose(posterior.sum(axis=1), 1.0)

print(model.get_parameters())
print(model.predict(observations[:5]))
print(model.cdf([1.0, 3.0, 6.0]))
print(model.quantile(0.99))                # Whole-mixture percentile
print(model.quantile(0.99, component=0))   # One component's percentile
print(model.dmp(8.0))                     # P(X > 8)
print(model.aic(observations), model.bic(observations))
print(model.converged_, model.n_iter_)
```

### Fitting and diagnostics

- Initialization uses K-means with `n_init=10` and honors supplied parameter
  seeds. `cv_init` may be a positive scalar shared initially by all components
  or an array with one entry per component; unrestricted fitting estimates a
  separate dispersion for each component.
- The E-step computes posterior component probabilities in log space. The
  M-step uses weighted inverse Gaussian estimates, avoiding unconstrained
  optimization over positive parameters.
- `min_cv=1e-8` places an absolute lower bound on each estimated dispersion.
  This prevents a component from collapsing to a zero-width spike at an
  observation. If the bound is active, the mean is reoptimized with dispersion
  fixed at the bound. Choose the bound in the same units as the observations;
  rescaling the data requires rescaling it too.
  For unrestricted mixture fitting, supplied `cv_init` values must be at least
  `min_cv`; lower the bound explicitly if smaller initial dispersions are
  intended. Fully specified models can still be evaluated with any positive
  dispersion. The fixed-dispersion RT mixture does not apply this floor.
- `max_iter` is the iteration budget. `n_iter_` is the number of completed EM
  iterations. `converged_` reports whether the stopping criterion was met;
  reaching the budget produces a `RuntimeWarning` and leaves it false.
  Convergence is tested against `tol` using the absolute change in mean
  log-likelihood per observation.
- `log_likelihood_history_` contains the initial log-likelihood and each
  accepted iteration. `lower_bound_` is the final log-likelihood per
  observation. `score(X)` returns the **total** log-likelihood, and
  `score_samples(X)` returns pointwise log densities.
- EM depends on initialization and can reach different local solutions. Fit
  several seeds and compare their scores when this matters. Component labels
  have no intrinsic ordering.

For $\ell=\sum_i\log p(x_i)$ and $p$ fitted parameters, model-selection scores
use the minimization convention

$$
\mathrm{AIC}=2p-2\ell,\qquad
\mathrm{BIC}=p\log n-2\ell.
$$

Lower scores are preferred when comparing fits to the same observations. The
ordinary mixture counts $p=3K-1$: $K$ modes, $K$ dispersions, and $K-1$
independent weights.

`ks_test(sample)` returns SciPy's Kolmogorov–Smirnov result for the whole mixture
and does not refit it. Its usual p-value is nominal when model parameters were
estimated from those same observations. Calibration must account for estimation,
for example with a parametric bootstrap that refits each replicate [3].

## Response-time models

The response-time parameterization uses backlog $b>0$, utilization
$0\le u<1$, and dispersion $\gamma>0$:

$$
\mu=\frac{b}{1-u},\qquad
\theta=\sqrt{\mu^2+\frac94\gamma^2}-\frac32\gamma.
$$

`RTInvGauss(backlog=b, deviation=delta, utilization=u)` additionally defines
$\gamma=\delta^2/(1-u)^2$ and $\lambda=b^2/\delta^2$. The argument `deviation`
is this model parameter $\delta$; the standard deviation of $X$ remains
$\sqrt{\mu\gamma}$. Fitting a single `RTInvGauss` keeps utilization fixed and
updates the backlog and deviation consistently with its fitted distribution.

`RTInvGaussMixture` fixes a common dispersion and utilization, and estimates
component backlogs and weights:

```python
from rInverseGaussian import RTInvGaussMixture

truth = RTInvGaussMixture(
    n_components=2,
    cv_init=0.2,
    backlog_init=[0.8, 2.8],
    weights_init=[0.65, 0.35],
    utilization=0.4,
)
response_times = truth.sample(2_000, random_state=7)

rt_model = RTInvGaussMixture(
    n_components=2,
    cv_init=0.2,
    utilization=0.4,
    max_iter=300,
    tol=1e-6,
    random_state=7,
).fit(response_times)

print(rt_model.backlog_)
print(rt_model.get_parameters())
print(rt_model.dmp(6.0))
```

`cv_init` is fixed throughout this fit; an array is accepted only when every
component is given the same dispersion. `backlog_init` and `modes_init` are
both supported; if both are supplied, they must describe the same components.
The parameter count is $p=2K-1$ because utilization and dispersion are fixed.

With normalized observation weights $w_i$, fixed dispersion requires a
different mean update from unrestricted fitting. Writing
$A=\sum_i w_i/x_i$, differentiation gives

$$
A\widehat\mu^2-\widehat\mu-\gamma=0,\qquad
\widehat\mu=\frac{1+\sqrt{1+4\gamma A}}{2A},\qquad
\widehat b=(1-u)\widehat\mu.
$$

The implementation evaluates an algebraically equivalent expression for
numerical stability. This update also applies to a one-component RT mixture.

For either mixture class, `dmp(deadline)` is exactly the fitted survival
probability

$$
P(X>d)=\sum_k\pi_k\,P(X_k>d).
$$

This is a model-based estimate of a deadline-miss probability. Its validity for
a real system depends on the model and data; fitting a distribution does not
establish a deterministic response-time bound.

## Input and compatibility notes

- Fitting requires finite, strictly positive observations. Invalid parameters
  and incompatible array lengths raise explicit exceptions.
  Mixtures require at least `n_components` observations; automatic K-means
  initialization also requires that many distinct observed values.
- Probability functions respect the positive support: at $x\le0$, the density
  and CDF are zero and the survival probability is one. Probability endpoints
  are supported: `quantile(0)` is zero and `quantile(1)` is positive infinity.
- `random_state` accepts a seed or a NumPy random-number generator. Omitting
  it retains support for existing code that uses `np.random.seed`.
- Existing public class names and legacy parameter names are retained. Known
  legacy optimizer names (`dogleg`, `BFGS`, `minimize_scalar`, and `newton`)
  passed through `method` are deprecated compatibility
  aliases; fitting now uses the closed-form updates regardless of that alias.
  Omit `method` in new code.
- Some returned values intentionally change because earlier implementations
  were incorrect: `cdf` returns the summed mixture CDF, `predict_proba` returns
  normalized posterior probabilities, and `dmp` evaluates a survival function.
  AIC/BIC use the lower-is-better convention above, and the single-distribution
  estimate of `cv` now uses dispersion rather than shape.

## Development

The repository keeps the original module names:

| File | Purpose |
| --- | --- |
| `rInvGauss.py` | Single distributions, likelihoods, sampling, kernel estimator |
| `rInvGaussMixture.py` | General mixture, EM, posterior probabilities, diagnostics |
| `RTInvGaussMixture.py` | Fixed-dispersion response-time mixture |
| `tests/` | Numerical and behavioral regression tests |
| `pyproject.toml` | Installation metadata and runtime dependencies |

Run the test suite from the repository root after installing the package:

```bash
python -m unittest discover -s tests -v
```

## References

1. Punzo, A. (2019). *A new look at the inverse Gaussian distribution with
   applications to insurance and economic data*. **Journal of Applied
   Statistics, 46**(7), 1260–1287.
   [doi:10.1080/02664763.2018.1542668](https://doi.org/10.1080/02664763.2018.1542668).
   The [2017 preprint](https://arxiv.org/abs/1707.04400) provides the distribution,
   parameter transformation, kernel estimator, and mixture construction.
   This implementation uses direct likelihood updates rather than the
   paper's numerical BFGS optimization.
2. SciPy developers. [`scipy.stats.invgauss` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html).
3. SciPy developers. [`scipy.stats.goodness_of_fit` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html),
   discussing goodness-of-fit tests with estimated distribution parameters.

