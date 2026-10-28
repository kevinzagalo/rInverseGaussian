"""Regression tests for the inverse Gaussian parameterization and numerics."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import invgauss, norm

from rInverseGaussian.rInvGauss import RTInvGauss, rInvGauss


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.model = rInvGauss(mode=2.0, cv=0.7)

    def test_parameter_conversion_and_moments(self):
        mu = np.sqrt(2.0 * (2.0 + 3 * 0.7))
        shape = mu**2 / 0.7
        self.assertAlmostEqual(self.model.mean, mu)
        self.assertAlmostEqual(self.model.shape, shape)
        self.assertAlmostEqual(self.model.variance, mu**3 / shape)
        self.assertAlmostEqual(self.model._mode(mu, shape), 2.0)
        self.assertAlmostEqual(self.model._cv(mu, shape), 0.7)
        modes = np.array([0.01, 1.0, 100.0])
        gammas = np.array([10.0, 0.1, 5.0])
        means = self.model._mean(modes, gammas)
        shapes = self.model._shape(modes, gammas)
        assert_allclose(self.model._mode(means, shapes), modes, rtol=1e-14)

    def test_mode_conversion_avoids_cancellation(self):
        mode = self.model._mode(mean=1.0, shape=1e-20)
        self.assertGreater(mode, 0.0)
        self.assertAlmostEqual(mode / (1e-20 / 3), 1.0, places=14)
        self.assertAlmostEqual(
            self.model._mode(mean=1e-10, shape=1e-320) / (1e-320 / 3),
            1.0, places=3,
        )
        mode = rInvGauss._mode_from_mean_dispersion(0.01, 1e307)
        self.assertGreater(mode, 0)
        self.assertAlmostEqual(mode / (1e-311 / 3), 1.0, places=10)
        self.assertAlmostEqual(rInvGauss._mean(mode, 1e307), 0.01, places=12)
        assert_allclose(
            rInvGauss._mode_from_mean_dispersion([1, 2], [0.1, 4]),
            rInvGauss._mode([1, 2], [10, 1]), rtol=1e-14,
        )

    def test_density_and_cdf_against_independent_formula(self):
        x = np.array([0.4, 1.0, 2.0, 3.5, 8.0])
        mu = np.sqrt(8.2)
        lam = 8.2 / 0.7
        density = np.sqrt(lam / (2 * np.pi * x**3)) * np.exp(
            -lam * (x - mu) ** 2 / (2 * mu**2 * x)
        )
        cdf = norm.cdf(np.sqrt(lam / x) * (x / mu - 1)) + np.exp(
            2 * lam / mu
        ) * norm.cdf(-np.sqrt(lam / x) * (x / mu + 1))
        assert_allclose(self.model.pdf(x), density, rtol=2e-14)
        assert_allclose(self.model.log_pdf(x), np.log(density), rtol=2e-14)
        assert_allclose(self.model.cdf(x), cdf, rtol=2e-14)
        integral, _ = quad(self.model.pdf, 0, np.inf, epsabs=1e-10)
        self.assertAlmostEqual(integral, 1.0, places=9)

    def test_support_boundaries_and_nan(self):
        x = np.array([-np.inf, -1.0, 0.0, np.inf, np.nan])
        assert_allclose(self.model.pdf(x), [0, 0, 0, 0, np.nan])
        assert_allclose(self.model.cdf(x), [0, 0, 0, 1, np.nan])
        assert_allclose(self.model.sf(x), [1, 1, 1, 0, np.nan])
        assert_allclose(self.model.log_pdf(x), [-np.inf] * 4 + [np.nan])
        assert_allclose(self.model.logsf(x), [0, 0, 0, -np.inf, np.nan])
        self.assertTrue(np.isscalar(self.model.pdf(1.0)))
        self.assertEqual(self.model.pdf(np.ones((2, 3))).shape, (2, 3))

    def test_tail_evaluation_does_not_subtract_cdf(self):
        deadline = 100.0
        self.assertEqual(self.model.cdf(deadline), 1.0)
        self.assertGreater(self.model.sf(deadline), 0.0)
        self.assertTrue(np.isfinite(self.model.logsf(deadline)))
        self.assertAlmostEqual(
            np.log(self.model.sf(deadline)), self.model.logsf(deadline), places=12
        )
        self.assertTrue(np.isfinite(self.model.logsf(1e5)))

    def test_quantiles(self):
        probabilities = np.array([1e-8, 0.01, 0.5, 0.99, 1 - 1e-8])
        quantiles = self.model.quantile(probabilities)
        self.assertTrue(np.all(np.diff(quantiles) > 0))
        assert_allclose(self.model.cdf(quantiles), probabilities, rtol=1e-10)
        assert_allclose(self.model.quantile([0, 1, np.nan]), [0, np.inf, np.nan])
        for invalid in [-0.01, 1.01, np.inf, -np.inf]:
            with self.subTest(alpha=invalid), self.assertRaises(ValueError):
                self.model.quantile(invalid)

    def test_extreme_quantiles_recover_from_inaccurate_scipy_inverse(self):
        model = rInvGauss(mode=1, cv=0.15)
        probabilities = np.array([1e-100, 1e-30, 1e-12, 0.5, 1 - 1e-12])
        quantiles = model.quantile(probabilities)
        self.assertTrue(np.all(np.isfinite(quantiles)))
        self.assertTrue(np.all(np.diff(quantiles) > 0))
        assert_allclose(model.cdf(quantiles[:-1]), probabilities[:-1], rtol=1e-11, atol=0)
        assert_allclose(model.sf(quantiles[-1]), 1 - probabilities[-1], rtol=1e-11, atol=0)
        self.assertGreater(model.quantile(1e-30), 0)

    def test_explicit_parameters_and_partial_overrides(self):
        assert_allclose(
            rInvGauss().pdf([1, 2], mode=2, cv=0.7), self.model.pdf([1, 2])
        )
        assert_allclose(
            self.model.pdf([1, 2], cv=0.2), rInvGauss(2, 0.2).pdf([1, 2])
        )
        with self.assertRaises(ValueError):
            self.model.pdf(1, mode=0)
        with self.assertRaises(ValueError):
            rInvGauss().pdf(1)

    def test_exact_mle_matches_conventional_parameter_estimate(self):
        sample = np.array([0.8, 1.2, 1.7, 2.3, 3.5, 5.0, 8.1])
        model = rInvGauss().fit(sample)
        mu = sample.mean()
        lam = 1 / np.mean(1 / sample - 1 / mu)
        self.assertAlmostEqual(model.mean, mu, places=13)
        self.assertAlmostEqual(model.shape, lam, places=13)
        self.assertAlmostEqual(model.cv, mu**2 / lam, places=13)
        scipy_mu, scipy_loc, scipy_scale = invgauss.fit(sample, floc=0)
        self.assertEqual(scipy_loc, 0)
        self.assertAlmostEqual(model.shape, scipy_scale, places=12)
        self.assertAlmostEqual(model.mean / model.shape, scipy_mu, places=12)
        assert_allclose(np.sum(model._dlogf(sample), axis=1), 0, atol=1e-13)
        self.assertTrue(model.converged_ and model.fitted_)
        self.assertEqual(model.n_iter_, 1)
        self.assertIs(model.fit(sample[:, None]), model)
        self.assertAlmostEqual(model.score(sample), model.log_pdf(sample).sum())

    def test_analytic_derivatives_against_finite_differences(self):
        parameters = np.array([2.0, 0.7])
        h = 1e-5
        for x in [0.5, 2.0, 7.0]:
            with self.subTest(x=x):
                gradient = np.empty(2)
                hessian = np.empty((2, 2))
                for index in range(2):
                    step = np.zeros(2)
                    step[index] = h
                    plus, minus = parameters + step, parameters - step
                    gradient[index] = (
                        self.model.log_pdf(x, *plus)
                        - self.model.log_pdf(x, *minus)
                    ) / (2 * h)
                    hessian[:, index] = (
                        self.model._dlogf(x, *plus)
                        - self.model._dlogf(x, *minus)
                    ) / (2 * h)
                assert_allclose(self.model._dlogf(x), gradient, rtol=2e-8, atol=1e-9)
                assert_allclose(self.model._hesslogf(x), hessian, rtol=2e-8, atol=1e-9)

    def test_invalid_parameters_and_samples(self):
        for value in [0, -1, np.inf, np.nan, True, [1, 2]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                rInvGauss(mode=value, cv=1)
            with self.subTest(cv=value), self.assertRaises(ValueError):
                rInvGauss(mode=1, cv=value)
        invalid_samples = [[], [1], [2, 2, 2], [0, 1], [-1, 2], [1, np.nan],
                           [1, np.inf], [[1, 2], [3, 4]], 2]
        for sample in invalid_samples:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                rInvGauss().fit(sample)
        for limit in [0, -1, 1.5, True]:
            with self.subTest(max_iter=limit), self.assertRaises(ValueError):
                rInvGauss(max_iter=limit)

    def test_kernel_density_preserves_query_shape(self):
        kernel = self.model.kde([1.0, 3.0, 5.0], cv=0.2)
        points = np.array([[0.5, 1.0], [3.0, 5.0]])
        expected = sum(rInvGauss(x, 0.2).pdf(points) for x in [1, 3, 5]) / 3
        assert_allclose(kernel(points), expected)
        self.assertTrue(np.isscalar(kernel(1.0)))
        self.assertEqual(kernel(np.array([])).shape, (0,))

    def test_sampling_reproducibility_and_moments(self):
        first = self.model.sample(20, random_state=7)
        assert_array_equal(first, self.model.sample(20, random_state=7))
        self.assertTrue(np.all(first > 0))
        state = np.random.get_state()
        try:
            np.random.seed(11)
            first = self.model.sample(20)
            np.random.seed(11)
            assert_array_equal(first, self.model.sample(20))
        finally:
            np.random.set_state(state)
        samples = self.model.sample(100000, random_state=42)
        standard_error = np.sqrt(self.model.variance / len(samples))
        self.assertLess(abs(samples.mean() - self.model.mean), 6 * standard_error)
        self.assertLess(abs(samples.var() / self.model.variance - 1), 0.05)
        for count in [0, -1, 1.5, True]:
            with self.subTest(n_sample=count), self.assertRaises(ValueError):
                self.model.sample(count)


class RTDistributionTests(unittest.TestCase):
    def test_physical_parameter_mapping(self):
        model = RTInvGauss(backlog=3, deviation=0.8, utilization=0.6)
        self.assertAlmostEqual(model.mean, 3 / 0.4)
        self.assertAlmostEqual(model.shape, 3**2 / 0.8**2)
        self.assertAlmostEqual(model.cv, 0.8**2 / 0.4**2)
        self.assertEqual(model.get_parameters()["backlog"], 3)

    def test_fit_updates_physical_parameters(self):
        model = RTInvGauss(utilization=0.6).fit([0.8, 1.2, 2.0, 4.5])
        reconstructed = RTInvGauss(
            backlog=model.backlog, deviation=model.deviation,
            utilization=model.utilization,
        )
        self.assertAlmostEqual(model.mode, reconstructed.mode)
        self.assertAlmostEqual(model.cv, reconstructed.cv)
        assert_allclose(model.pdf([1, 2, 4]), reconstructed.pdf([1, 2, 4]))

    def test_invalid_physical_parameters(self):
        for utilization in [None, -0.1, 1, np.inf, np.nan, [0.2, 0.3]]:
            with self.subTest(u=utilization), self.assertRaises(ValueError):
                RTInvGauss(1, 1, utilization)
        for kwargs in [{"backlog": 0, "deviation": 1},
                       {"backlog": 1, "deviation": 0},
                       {"backlog": 1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RTInvGauss(utilization=0.4, **kwargs)


if __name__ == "__main__":
    unittest.main()
