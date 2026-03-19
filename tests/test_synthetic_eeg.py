"""Tests for SyntheticEEGGenerator."""

import numpy as np
import pytest

from bci2token.synthetic_eeg import SyntheticEEGGenerator, MENTAL_STATES, STATE_PROFILES


class TestSyntheticEEGGenerator:
    def setup_method(self):
        self.gen = SyntheticEEGGenerator(
            n_channels=8, fs=256.0, duration_s=2.0, rng_seed=99
        )

    def test_all_states_generate(self):
        for state in MENTAL_STATES:
            trial = self.gen.generate(state)
            assert trial.state == state

    def test_output_shape(self):
        trial = self.gen.generate("rest")
        expected_samples = int(256.0 * 2.0)
        assert trial.data.shape == (8, expected_samples)

    def test_dtype_float32(self):
        trial = self.gen.generate("rest")
        assert trial.data.dtype == np.float32

    def test_different_states_differ(self):
        """Different mental states should produce distinct EEG profiles."""
        t1 = self.gen.generate("rest")
        t2 = self.gen.generate("mental_arithmetic")
        # Signals should differ (different spectral profiles)
        assert not np.allclose(t1.data, t2.data, atol=1e-3)

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            self.gen.generate("daydreaming")

    def test_generate_batch_shape(self):
        batch = self.gen.generate_batch("visual_attention", n_trials=5)
        assert batch.shape == (5, 8, int(256.0 * 2.0))

    def test_generate_dataset(self):
        X, y = self.gen.generate_dataset(n_per_state=3)
        n_states = len(MENTAL_STATES)
        assert X.shape[0] == 3 * n_states
        assert len(y) == 3 * n_states
        # All states represented
        for state in MENTAL_STATES:
            assert state in y

    def test_reproducibility(self):
        gen_a = SyntheticEEGGenerator(n_channels=4, fs=128.0, duration_s=1.0, rng_seed=42)
        gen_b = SyntheticEEGGenerator(n_channels=4, fs=128.0, duration_s=1.0, rng_seed=42)
        t_a = gen_a.generate("rest")
        t_b = gen_b.generate("rest")
        np.testing.assert_array_equal(t_a.data, t_b.data)

    def test_channels_have_different_signals(self):
        """Frontal and occipital channels should differ due to spatial modulation."""
        trial = self.gen.generate("visual_attention")
        # Frontal ch 0 vs occipital ch 7 should differ
        assert not np.allclose(trial.data[0], trial.data[7], atol=1e-3)

    def test_trial_properties(self):
        trial = self.gen.generate("rest")
        assert trial.n_channels == 8
        assert trial.n_timesteps == int(256.0 * 2.0)
        assert trial.fs == 256.0
        assert trial.duration_s == 2.0


class TestSpectralProperties:
    """Verify that generated EEG has plausible spectral content per state."""

    def _band_power(self, signal, fs, fmin, fmax):
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = np.abs(np.fft.rfft(signal)) ** 2 / n
        return np.sum(psd[(freqs >= fmin) & (freqs <= fmax)])

    def test_rest_has_high_alpha(self):
        gen = SyntheticEEGGenerator(n_channels=4, fs=256.0, duration_s=4.0, rng_seed=1)
        # Average over channels
        alpha_powers, theta_powers = [], []
        for _ in range(10):
            trial = gen.generate("rest")
            for ch in range(trial.n_channels):
                alpha_powers.append(self._band_power(trial.data[ch], 256.0, 8, 13))
                theta_powers.append(self._band_power(trial.data[ch], 256.0, 4, 8))
        assert np.mean(alpha_powers) > np.mean(theta_powers), \
            "Rest state should have more alpha than theta power"

    def test_arithmetic_has_high_theta(self):
        gen = SyntheticEEGGenerator(n_channels=4, fs=256.0, duration_s=4.0, rng_seed=2)
        frontal_theta, frontal_alpha = [], []
        for _ in range(10):
            trial = gen.generate("mental_arithmetic")
            # Frontal channels only (first quarter)
            for ch in range(1):
                frontal_theta.append(self._band_power(trial.data[ch], 256.0, 4, 8))
                frontal_alpha.append(self._band_power(trial.data[ch], 256.0, 8, 13))
        assert np.mean(frontal_theta) > np.mean(frontal_alpha), \
            "Mental arithmetic should elevate frontal theta over alpha"
