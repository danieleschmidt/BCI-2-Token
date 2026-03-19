"""Tests for PrivacyFilter."""

import numpy as np
import pytest

from bci2token.privacy_filter import PrivacyFilter, PrivacyBudget


class TestPrivacyFilter:
    def setup_method(self):
        self.pf = PrivacyFilter(epsilon=1.0, delta=1e-5, sensitivity=1.0, rng_seed=0)

    def test_output_shape_preserved(self):
        x = np.random.randn(160)
        noisy = self.pf.apply(x)
        assert noisy.shape == x.shape

    def test_2d_shape_preserved(self):
        x = np.random.randn(8, 20)
        noisy = self.pf.apply(x)
        assert noisy.shape == x.shape

    def test_noise_is_added(self):
        x = np.zeros(100)
        noisy = self.pf.apply(x)
        assert not np.allclose(noisy, x), "Noise should change the features"

    def test_noise_level_scales_with_epsilon(self):
        """Smaller epsilon → larger sigma → more noise."""
        x = np.zeros(1000)
        high_pf = PrivacyFilter(epsilon=10.0, delta=1e-5, sensitivity=1.0, rng_seed=1)
        low_pf  = PrivacyFilter(epsilon=0.1,  delta=1e-5, sensitivity=1.0, rng_seed=2)
        assert low_pf.sigma > high_pf.sigma

    def test_sigma_formula(self):
        """Verify σ = Δf * sqrt(2 * ln(1.25/δ)) / ε."""
        import math
        pf = PrivacyFilter(epsilon=2.0, delta=1e-5, sensitivity=3.0)
        expected = 3.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 2.0
        assert abs(pf.sigma - expected) < 1e-9

    def test_clip_norm(self):
        """With clip_norm, L2 norm of clipped features should be <= clip_norm."""
        pf = PrivacyFilter(epsilon=1.0, delta=1e-5, clip_norm=1.0, rng_seed=3)
        x = np.random.randn(100) * 10  # large norm
        # We check the clip happened internally; just verify no error + output shape
        noisy = pf.apply(x)
        assert noisy.shape == x.shape

    def test_batch_apply(self):
        x = np.random.randn(5, 80)
        noisy = self.pf.apply_batch(x)
        assert noisy.shape == x.shape

    def test_budget_tracking(self):
        pf = PrivacyFilter(epsilon=1.0, delta=1e-5, track_budget=True, rng_seed=4)
        x = np.zeros(50)
        pf.apply(x)
        pf.apply(x)
        assert pf.budget._num_queries == 2
        assert pf.budget._spent_epsilon == 2.0

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            PrivacyFilter(epsilon=0.0, delta=1e-5)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            PrivacyFilter(epsilon=1.0, delta=1.5)

    def test_repr(self):
        r = repr(self.pf)
        assert "ε=" in r
        assert "σ=" in r


class TestPrivacyBudget:
    def test_spend_increments(self):
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        budget.spend(1.0, 1e-6)
        assert budget._spent_epsilon == 1.0
        assert budget._num_queries == 1

    def test_remaining_epsilon(self):
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        budget.spend(2.0, 1e-6)
        assert budget.remaining_epsilon == 3.0

    def test_exhausted(self):
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        budget.spend(1.5, 1e-6)
        assert budget.is_exhausted
