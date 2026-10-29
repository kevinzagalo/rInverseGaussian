"""Scientific and API regression tests for the fixed-dispersion RT mixture."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar
from scipy.stats import invgauss

from rInverseGaussian.RTInvGaussMixture import RTInvGaussMixture


class RealTimeMixtureTests(unittest.TestCase):
    def make_model(self):
        return RTInvGaussMixture(
            2, cv_init=0.2, backlog_init=[0.8, 2.8],
            weights_init=[0.65, 0.35], utilization=0.4, random_state=7,
        )

    def test_physical_conversion_and_preconfigured_sampling(self):
        model = self.make_model()
        assert_allclose(model._means, np.array([0.8, 2.8]) / 0.6)
        assert_allclose(model._backlog(model.modes_, 0.2), [0.8, 2.8])
        assert_allclose(model._mode(model.backlog_), model.modes_)
        first = model.sample(12, random_state=13)
        assert_allclose(first, model.sample(12, random_state=13))
        self.assertEqual(first.shape, (12,))
        self.assertTrue(np.all(first > 0))

    def test_constructor_validation(self):
        for utilization in [None, -0.01, 1, 2, np.nan, np.inf, True]:
            with self.subTest(utilization=utilization), self.assertRaises(ValueError):
                RTInvGaussMixture(2, cv_init=0.2, utilization=utilization)
        for dispersion in [None, 0, -1, np.nan, np.inf, True, np.bool_(True),
                           [True, True], [0.2], [0.2, 0.3], [[0.2, 0.2]]]:
            with self.subTest(dispersion=dispersion), self.assertRaises(ValueError):
                RTInvGaussMixture(2, cv_init=dispersion, utilization=0.5)
        for backlog in [0.2, [], [1], [1, -1], [1, np.nan]]:
            with self.subTest(backlog=backlog), self.assertRaises(ValueError):
                RTInvGaussMixture(2, cv_init=0.2, utilization=0.5, backlog_init=backlog)
        model = RTInvGaussMixture(2, cv_init=np.array([0.2, 0.2]), utilization=0)
        assert_allclose(model.cv_, [0.2, 0.2])

    def test_physical_conversion_avoids_intermediate_overflow(self):
        model = RTInvGaussMixture(1, 1e307, backlog_init=[0.01], utilization=0)
        self.assertGreater(model.modes_[0], 0)
        assert_allclose(model.backlog_, [0.01], rtol=1e-10)
        assert_allclose(model.dtheta_dbacklog(0.01), 0.01 / 1e307 / 1.5,
                        rtol=1e-12, atol=0)
        assert_allclose(model.d2theta_dbacklog2(0.01), 1 / 1e307 / 1.5,
                        rtol=1e-12, atol=0)
        model = RTInvGaussMixture(1, 1e308, modes_init=[1], utilization=0)
        assert_allclose(model.backlog_, [np.sqrt(3) * 1e154])

    def test_initialization_honors_backlogs_modes_and_weights(self):
        source = self.make_model()
        model = RTInvGaussMixture(
            2, cv_init=0.2, utilization=0.4,
            modes_init=source.modes_, backlog_init=[0.8, 2.8],
            weights_init=[0.65, 0.35],
        )
        model.initialize(np.array([0.5, 0.8, 1.2, 2.1, 4, 8]))
        assert_allclose(model.modes_, source.modes_)
        assert_allclose(model.backlog_, [0.8, 2.8])
        assert_allclose(model.weights_, [0.65, 0.35])
        with self.assertRaisesRegex(ValueError, "different components"):
            RTInvGaussMixture(
                2, cv_init=0.2, utilization=0.4,
                modes_init=source.modes_ * 2, backlog_init=[0.8, 2.8],
            )

    def test_weighted_m_step_matches_independent_numerical_optimization(self):
        data = np.array([0.35, 0.8, 1.2, 2.6, 5.2])
        weights = np.array([1, 0.5, 0.2, 0, 2])
        gamma, utilization = 0.65, 0.45
        model = RTInvGaussMixture(1, gamma, utilization=utilization)

        def objective(log_mean):
            mean = np.exp(log_mean)
            shape = mean ** 2 / gamma
            return -np.dot(weights, invgauss.logpdf(data, mean / shape, scale=shape))

        optimum = minimize_scalar(objective, bounds=(-5, 5), method="bounded",
                                  options={"xatol": 1e-12})
        self.assertTrue(optimum.success)
        backlog = model._update_params(data, weights)
        mean = backlog / (1 - utilization)
        assert_allclose(mean, np.exp(optimum.x), rtol=1e-7)
        reciprocal_mean = np.dot(weights, 1 / data) / weights.sum()
        self.assertAlmostEqual(reciprocal_mean * mean ** 2 - mean - gamma, 0, places=12)

    def test_weighted_m_step_rejects_unknown_optimization_method(self):
        model = RTInvGaussMixture(1, 0.2, utilization=0.5)
        with self.assertRaisesRegex(ValueError, "method"):
            model._update_params([1, 2, 3], [1, 1, 1], method="unknown")

    def test_single_component_uses_fixed_dispersion_mle(self):
        data = np.array([0.4, 0.8, 1.2, 2.3, 4.8])
        gamma, utilization = 0.65, 0.45
        model = RTInvGaussMixture(1, gamma, utilization=utilization).fit(data)
        reciprocal_mean = np.mean(1 / data)
        expected = (1 + np.sqrt(1 + 4 * gamma * reciprocal_mean)) / (2 * reciprocal_mean)
        assert_allclose(model._means[0], expected, rtol=1e-12)
        self.assertGreater(abs(model._means[0] - data.mean()), 0.1)
        assert_allclose(model.cv_, [gamma])
        assert_allclose(model.weights_, [1])
        self.assertTrue(model.fitted_)
        self.assertTrue(model.converged_)

    def test_fixed_dispersion_has_finite_mle_for_constant_data(self):
        model = RTInvGaussMixture(1, 0.3, utilization=0.2).fit([2, 2, 2])
        expected = (2 + np.sqrt(4 + 4 * 0.3 * 2)) / 2
        assert_allclose(model._means, [expected], rtol=1e-12)
        self.assertTrue(np.isfinite(model.score([2, 2, 2])))

    def test_fixed_dispersion_below_base_floor_is_preserved(self):
        model = RTInvGaussMixture(1, 1e-12, utilization=0.2).fit([1, 1.01, 1.02])
        assert_allclose(model.cv_, [1e-12], rtol=0, atol=0)

    def test_deadline_miss_probability_is_exact_mixture_survival(self):
        model = self.make_model()
        deadlines = np.array([-1, 0, 0.5, 2, 10, 100, np.inf])
        expected = np.zeros_like(deadlines)
        for weight, mean in zip(model.weights_, model._means):
            shape = mean ** 2 / 0.2
            expected += weight * invgauss.sf(deadlines, mean / shape, scale=shape)
        assert_allclose(model.dmp(deadlines), expected, rtol=1e-12)
        single = RTInvGaussMixture(
            1, 0.2, backlog_init=[0.8], weights_init=[1], utilization=0.4,
        )
        self.assertGreater(single.dmp(single._means[0]), 0.1)

    def test_em_preserves_dispersion_and_likelihood_monotonicity(self):
        data = self.make_model().sample(800, random_state=19)
        model = RTInvGaussMixture(
            2, 0.2, utilization=0.4, random_state=11, max_iter=200, tol=1e-7,
        ).fit(data)
        assert_allclose(model.cv_, [0.2, 0.2], rtol=0, atol=0)
        assert_allclose(model.backlog_, model._backlog(model.modes_, 0.2))
        assert_allclose(model.weights_.sum(), 1)
        assert_allclose(model.predict_proba(data).sum(axis=1), 1, atol=1e-12)
        self.assertTrue(np.isfinite(model.score(data)))
        self.assertTrue(np.all(np.diff(model.log_likelihood_history_) >= -1e-7))

    def test_information_criteria_count_only_estimated_parameters(self):
        model = self.make_model()
        data = np.array([0.5, 0.8, 1.2, 2.1, 4, 8])
        log_likelihood = model.score(data)
        n_parameters = 2 * 2 - 1
        self.assertAlmostEqual(model.aic(data), 2 * n_parameters - 2 * log_likelihood)
        self.assertAlmostEqual(model.bic(data), np.log(data.size) * n_parameters - 2 * log_likelihood)

    def test_backlog_derivatives_match_finite_differences(self):
        model = self.make_model()
        for backlog in [0.1, 0.8, 3]:
            step = 1e-5
            first = (model._mode(backlog + step) - model._mode(backlog - step)) / (2 * step)
            second = (model.dtheta_dbacklog(backlog + step)
                      - model.dtheta_dbacklog(backlog - step)) / (2 * step)
            assert_allclose(model.dtheta_dbacklog(backlog), first, rtol=1e-8)
            assert_allclose(model.d2theta_dbacklog2(backlog), second, rtol=1e-8)

    def test_parameter_roundtrip_and_failed_restore_are_consistent(self):
        source = self.make_model()
        params = source.get_parameters()
        restored = RTInvGaussMixture(2, 1, utilization=0.1)
        self.assertIs(restored.set_parameters(params), restored)
        self.assertEqual(restored.get_parameters(), params)
        assert_allclose(restored.pdf([0.1, 1, 5]), source.pdf([0.1, 1, 5]))
        restored.initialize(np.array([0.5, 0.8, 1.2, 2.1, 4, 8]))
        assert_allclose(restored.cv_, source.cv_)
        assert_allclose(restored.backlog_, source.backlog_)
        invalid = {**params, "utilization": 2}
        with self.assertRaises(ValueError):
            restored.set_parameters(invalid)
        self.assertEqual(restored.get_parameters(), params)


if __name__ == "__main__":
    unittest.main()
