"""Regression tests for mixture probabilities, likelihood estimation, and state."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import invgauss, kstest

from rInverseGaussian.rInvGauss import rInvGauss
from rInverseGaussian.rInvGaussMixture import (
    _fixed_dispersion_mean,
    rInvGaussMixture,
)


class MixtureTests(unittest.TestCase):
    def make_model(self, **kwargs):
        return rInvGaussMixture(
            2, modes_init=[1.0, 4.0], cv_init=[0.15, 0.6],
            weights_init=[0.3, 0.7], **kwargs,
        )

    @staticmethod
    def scipy_components():
        # Independently convert the specified modes and dispersions.
        means = np.sqrt(np.array([1.0, 4.0]) * np.array([1.45, 5.8]))
        shapes = means**2 / np.array([0.15, 0.6])
        return [invgauss(mean / shape, scale=shape)
                for mean, shape in zip(means, shapes)]

    def test_mixture_density_cdf_and_survival_against_scipy(self):
        model = self.make_model()
        values = np.array([-np.inf, 0, 0.4, 1, 2.5, 5, 12, 100, np.inf])
        components = self.scipy_components()
        for method in ["pdf", "cdf", "sf"]:
            expected = sum(weight * getattr(component, method)(values)
                           for weight, component in zip([0.3, 0.7], components))
            assert_allclose(getattr(model, method)(values), expected, rtol=1e-12)
        assert_allclose(model.dmp(values), model.sf(values), rtol=0, atol=0)
        self.assertEqual(model.cdf(100), 1.0)
        self.assertGreater(model.sf(100), 0)
        self.assertTrue(np.isfinite(model.logsf(100)))
        self.assertTrue(np.isscalar(model.cdf(2)))
        self.assertEqual(model.pdf(np.ones((2, 3))).shape, (2, 3))

    def test_density_integrates_to_one(self):
        integral, error = quad(self.make_model().pdf, 0, np.inf, epsabs=1e-10)
        self.assertLess(error, 1e-8)
        self.assertAlmostEqual(integral, 1, places=9)

    def test_posterior_probabilities_remain_normalized_when_pdf_underflows(self):
        model = self.make_model()
        values = np.array([1e-5, 0.8, 2, 7, 1e5])
        reference_joint = np.column_stack([
            np.log(weight) + component.logpdf(values)
            for weight, component in zip([0.3, 0.7], self.scipy_components())
        ])
        expected = np.exp(reference_joint - logsumexp(reference_joint, axis=1)[:, None])
        probabilities = model.predict_proba(values)
        assert_allclose(probabilities, expected, rtol=1e-10, atol=1e-14)
        assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14)
        self.assertEqual(model.pdf(values[0]), 0)
        self.assertEqual(model.pdf(values[-1]), 0)
        self.assertTrue(np.all(np.isfinite(model.score_samples(values))))
        assert_array_equal(model.predict(values), probabilities.argmax(axis=1))

    def test_identical_components_retain_prior_weights_in_extreme_tail(self):
        model = rInvGaussMixture(
            2, modes_init=[1, 1], cv_init=[0.5, 0.5], weights_init=[0.2, 0.8],
        )
        # Adding O(1) log weights directly to a log density near -1e100 loses
        # the weights. A shifted computation must recover the exact symmetry.
        assert_allclose(model.predict_proba([1, 1e100]), [[0.2, 0.8], [0.2, 0.8]],
                        rtol=1e-14, atol=1e-15)

    def test_zero_weight_components_have_zero_posterior(self):
        model = rInvGaussMixture(
            2, modes_init=[1, 4], cv_init=[0.15, 0.6], weights_init=[0, 1],
        )
        values = [1e-5, 1, 5, 1e5]
        assert_allclose(model.predict_proba(values), [[0, 1]] * len(values), atol=0)
        assert_allclose(model.pdf(values), self.scipy_components()[1].pdf(values))
        self.assertTrue(np.isfinite(model.score(values)))

    def test_mixture_and_component_quantiles(self):
        model = self.make_model()
        probabilities = np.array([1e-30, 1e-4, 0.2, 0.5, 0.95, 1 - 1e-12])
        quantiles = model.quantile(probabilities)
        self.assertTrue(np.all(np.diff(quantiles) > 0))
        assert_allclose(model.cdf(quantiles), probabilities, rtol=2e-9, atol=0)
        assert_allclose(model.sf(quantiles[-2:]), 1 - probabilities[-2:], rtol=2e-9)
        for component, reference in enumerate(self.scipy_components()):
            component_quantiles = model.quantile(probabilities, component=component)
            assert_allclose(reference.cdf(component_quantiles), probabilities,
                            rtol=2e-9, atol=0)
            assert_allclose(reference.sf(component_quantiles[-2:]),
                            1 - probabilities[-2:], rtol=2e-9, atol=0)
        assert_allclose(model.quantile([0, 1]), [0, np.inf])
        for probability in [-0.1, 1.1, np.nan, np.inf]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                model.quantile(probability)
        with self.assertRaises(ValueError):
            model.quantile(0.5, component=2)

    def test_weighted_mle_matches_independent_numerical_optimization(self):
        data = np.array([0.3, 0.6, 1.1, 2.4, 5.2, 8.0])
        weights = np.array([0.4, 1.8, 0.7, 0.05, 1.2, 0])

        def objective(parameters):
            mean, shape = np.exp(parameters)
            return -np.dot(weights, invgauss.logpdf(data, mean / shape, scale=shape))

        optimum = minimize(objective, np.log([1.5, 2]), method="Nelder-Mead",
                           options={"xatol": 1e-11, "fatol": 1e-11, "maxiter": 1000})
        self.assertTrue(optimum.success)
        mode, dispersion = rInvGaussMixture(1)._update_params(data, weights)
        fitted = rInvGauss(mode, dispersion)
        assert_allclose([fitted.mean, fitted.shape], np.exp(optimum.x), rtol=2e-7)
        self.assertAlmostEqual(objective(np.log([fitted.mean, fitted.shape])),
                               optimum.fun, places=10)

    def test_dispersion_floor_reoptimizes_the_mean(self):
        data = np.array([0.8, 0.9, 1.0, 1.2])
        weights = np.array([1, 2, 0.5, 1])
        floor = 0.4

        def objective(log_mean):
            mean = np.exp(log_mean)
            shape = mean**2 / floor
            return -np.dot(weights, invgauss.logpdf(data, mean / shape, scale=shape))

        optimum = minimize_scalar(objective, bounds=(-5, 5), method="bounded",
                                  options={"xatol": 1e-12})
        self.assertTrue(optimum.success)
        mode, dispersion = rInvGaussMixture(1, min_cv=floor)._update_params(data, weights)
        self.assertEqual(dispersion, floor)
        mean = rInvGauss(mode, dispersion).mean
        assert_allclose(mean, np.exp(optimum.x), rtol=1e-7)
        self.assertGreater(abs(mean - np.average(data, weights=weights)), 0.1)
        self.assertAlmostEqual(objective(np.log(mean)), optimum.fun, places=10)

    def test_fixed_dispersion_mean_handles_tiny_positive_observations(self):
        tiny = 1e-320
        result = _fixed_dispersion_mean(np.array([tiny]), np.array([1.0]), 1e-10)
        self.assertGreater(result, 0)
        assert_allclose(result, np.sqrt(tiny) * np.sqrt(1e-10), rtol=1e-12, atol=0)

    def test_single_component_fit_matches_scipy_maximum_likelihood(self):
        data = np.array([0.4, 0.7, 1.1, 1.5, 3.4, 6.2])
        model = rInvGaussMixture(1).fit(data)
        scipy_mu, _, scipy_shape = invgauss.fit(data, floc=0)
        fitted = rInvGauss(model.modes_[0], model.cv_[0])
        assert_allclose([fitted.mean, fitted.shape],
                        [scipy_mu * scipy_shape, scipy_shape], rtol=1e-12)
        assert_allclose(model.weights_, [1])
        self.assertTrue(model.converged_ and model.fitted_)

    def test_em_likelihood_monotonicity_and_reproducible_refit(self):
        data = self.make_model().sample(400, random_state=19)
        model = rInvGaussMixture(2, random_state=8, max_iter=200, tol=1e-7).fit(data)
        self.assertTrue(model.converged_ and model.fitted_)
        self.assertTrue(np.all(np.diff(model.log_likelihood_history_) >= -1e-8))
        self.assertEqual(len(model.log_likelihood_history_), model.n_iter_ + 1)
        self.assertAlmostEqual(model.lower_bound_, model.score(data) / len(data))
        parameters = model.get_parameters()
        history = list(model.log_likelihood_history_)
        model.fit(data)
        self.assertEqual(model.get_parameters(), parameters)
        assert_allclose(model.log_likelihood_history_, history, rtol=0, atol=0)

    def test_nonconverged_refit_resets_success_state(self):
        data = self.make_model().sample(400, random_state=11)
        model = self.make_model(max_iter=200, tol=1e-7).fit(data)
        self.assertTrue(model.converged_)
        model.max_iter, model.tol = 1, 0
        with self.assertWarnsRegex(RuntimeWarning, "max_iter"):
            model.fit(2 * data + 0.2)
        self.assertFalse(model.converged_)
        self.assertTrue(model.fitted_)
        self.assertEqual(model.n_iter_, 1)
        self.assertEqual(len(model.log_likelihood_history_), 2)

    def test_initialization_honors_inputs_without_aliasing(self):
        modes = np.array([0.8, 3.5])
        dispersions = np.array([0.2, 0.4])
        weights = np.array([0.4, 0.6])
        model = rInvGaussMixture(2, modes_init=modes, cv_init=dispersions,
                                weights_init=weights)
        data = np.array([0.5, 0.8, 1.1, 2.0, 4, 8])
        original_data = data.copy()
        model.initialize(data)
        assert_array_equal(data, original_data)
        assert_array_equal(model.modes_, modes)
        assert_array_equal(model.cv_, dispersions)
        assert_array_equal(model.weights_, weights)
        modes[:], dispersions[:], weights[:] = 99, 99, 0.5
        assert_array_equal(model.modes_, [0.8, 3.5])
        assert_array_equal(model.cv_, [0.2, 0.4])
        assert_array_equal(model.weights_, [0.4, 0.6])

    def test_initial_dispersion_below_floor_is_rejected_for_fitting(self):
        model = rInvGaussMixture(1, modes_init=[1], cv_init=[1e-12], weights_init=[1])
        self.assertTrue(np.isfinite(model.pdf(1)))
        with self.assertRaisesRegex(ValueError, "min_cv"):
            model.fit([1, 1, 1])
        self.assertFalse(model.fitted_)

    def test_total_likelihood_and_information_criteria(self):
        data = np.array([0.5, 0.9, 1.8, 3.0, 4.7, 8.0])
        log_joint = np.column_stack([
            np.log(weight) + component.logpdf(data)
            for weight, component in zip([0.3, 0.7], self.scipy_components())
        ])
        log_likelihood = logsumexp(log_joint, axis=1).sum()
        model = self.make_model()
        self.assertAlmostEqual(model.score(data), log_likelihood, places=12)
        self.assertAlmostEqual(np.log(model.likelihood(data)), log_likelihood, places=12)
        self.assertAlmostEqual(model.aic(data), 2 * 5 - 2 * log_likelihood, places=12)
        self.assertAlmostEqual(model.bic(data), np.log(len(data)) * 5 - 2 * log_likelihood,
                               places=12)

    def test_parameter_roundtrip_copies_arrays_and_preserves_initialization(self):
        source = self.make_model()
        parameters = source.get_parameters()
        expected = copy.deepcopy(parameters)
        restored = rInvGaussMixture(2)
        self.assertIs(restored.set_parameters(parameters), restored)
        assert_allclose(restored.pdf([0.4, 1, 3, 8]), source.pdf([0.4, 1, 3, 8]))
        parameters["modes"][0] = 100
        parameters["cv"][0] = 100
        parameters["weights"][0] = 0
        self.assertEqual(restored.get_parameters(), expected)
        restored.initialize([0.5, 1, 2, 3, 4, 6])
        self.assertEqual(restored.get_parameters(), expected)
        with self.assertRaises(ValueError):
            restored.set_parameters({**expected, "weights": [1, 1]})
        self.assertEqual(restored.get_parameters(), expected)

    def test_ks_diagnostic_does_not_fit_or_mutate_parameters(self):
        model = self.make_model()
        data = model.sample(60, random_state=18)
        before = model.get_parameters()
        components = self.scipy_components()

        def independent_cdf(values):
            return 0.3 * components[0].cdf(values) + 0.7 * components[1].cdf(values)

        reference = kstest(data, independent_cdf)
        with patch.object(model, "fit", side_effect=AssertionError("unexpected fit")):
            result = model.ks_test(data)
        assert_allclose([result.statistic, result.pvalue],
                        [reference.statistic, reference.pvalue], rtol=1e-12)
        self.assertEqual(model.get_parameters(), before)
        self.assertFalse(model.fitted_)

    def test_sampling_seed_and_mixture_mean(self):
        model = self.make_model()
        assert_array_equal(model.sample(30, random_state=12), model.sample(30, random_state=12))
        samples = model.sample(30000, random_state=32)
        self.assertTrue(np.all(samples > 0))
        components = self.scipy_components()
        mean = sum(w * c.mean() for w, c in zip([0.3, 0.7], components))
        variance = sum(w * (c.var() + c.mean()**2)
                       for w, c in zip([0.3, 0.7], components)) - mean**2
        self.assertLess(abs(samples.mean() - mean), 6 * np.sqrt(variance / len(samples)))
        for count in [0, -1, 1.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                model.sample(count)

    def test_invalid_constructor_parameters(self):
        for count in [0, -1, 1.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                rInvGaussMixture(count)
        for kwargs in [{"modes_init": [1]}, {"modes_init": [0, 1]},
                       {"cv_init": [1, np.nan]}, {"cv_init": 0},
                       {"weights_init": [0.4, 0.5]}, {"weights_init": [-1, 2]},
                       {"weights_init": [np.nan, 1]}, {"min_cv": 0},
                       {"tol": -1}, {"max_iter": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rInvGaussMixture(2, **kwargs)

    def test_invalid_samples_and_uninitialized_evaluation(self):
        for sample in [[], [1], [0, 1], [-1, 2], [1, np.nan], [1, np.inf],
                       [[1, 2], [3, 4]], 1]:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                rInvGaussMixture(2).fit(sample)
        with self.assertRaises(ValueError):
            rInvGaussMixture(2).fit([1, 1, 1])
        with self.assertRaises(ValueError):
            rInvGaussMixture(2).pdf([1, 2])
        with self.assertRaises(ValueError):
            self.make_model().predict_proba([0, 1])
        with self.assertRaises(ValueError):
            self.make_model().fit([1, 2], method="unknown")


if __name__ == "__main__":
    unittest.main()
