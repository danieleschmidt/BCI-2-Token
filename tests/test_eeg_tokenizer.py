"""Tests for EEGTokenizer."""

import numpy as np
import pytest

from bci2token.eeg_tokenizer import EEGTokenizer, extract_spectral_features, BANDS


class TestSpectralFeatures:
    def test_output_shape(self):
        n_channels, n_timesteps = 8, 512
        eeg = np.random.randn(n_channels, n_timesteps).astype(np.float32)
        feats = extract_spectral_features(eeg)
        assert feats.shape == (n_channels * len(BANDS),)

    def test_non_negative_after_log(self):
        eeg = np.random.randn(4, 256).astype(np.float32)
        feats = extract_spectral_features(eeg)
        # log1p(x) >= 0 for x >= 0
        assert np.all(feats >= 0), "Features should be non-negative after log1p"

    def test_different_states_produce_different_features(self):
        """Alpha-dominant and theta-dominant signals should yield different features."""
        fs = 256.0
        n = 512
        t = np.linspace(0, n / fs, n)
        # Channel 0: strong alpha (10 Hz), Channel 1: strong theta (6 Hz)
        eeg_alpha = np.zeros((2, n), dtype=np.float32)
        eeg_alpha[0] = 5.0 * np.sin(2 * np.pi * 10.0 * t)  # alpha

        eeg_theta = np.zeros((2, n), dtype=np.float32)
        eeg_theta[0] = 5.0 * np.sin(2 * np.pi * 6.0 * t)   # theta

        f_alpha = extract_spectral_features(eeg_alpha, fs=fs)
        f_theta = extract_spectral_features(eeg_theta, fs=fs)

        assert not np.allclose(f_alpha, f_theta, atol=1e-3), \
            "Alpha and theta signals should produce distinct features"

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            extract_spectral_features(np.random.randn(256))


class TestEEGTokenizer:
    def setup_method(self):
        self.tok = EEGTokenizer(n_channels=8, vocab_size=256, fs=256.0)

    def test_encode_output_shape(self):
        eeg = np.random.randn(8, 512).astype(np.float32)
        logits = self.tok.encode(eeg)
        assert logits.shape == (256,)

    def test_encode_returns_floats(self):
        eeg = np.random.randn(8, 512).astype(np.float32)
        logits = self.tok.encode(eeg)
        assert logits.dtype in (np.float32, np.float64)

    def test_encode_batch(self):
        batch = np.random.randn(4, 8, 512).astype(np.float32)
        logits = self.tok.encode_batch(batch)
        assert logits.shape == (4, 256)

    def test_different_trials_differ(self):
        eeg1 = np.random.randn(8, 512).astype(np.float32)
        eeg2 = np.random.randn(8, 512).astype(np.float32)
        l1 = self.tok.encode(eeg1)
        l2 = self.tok.encode(eeg2)
        assert not np.allclose(l1, l2, atol=1e-3), "Different EEG should yield different logits"

    def test_feature_names_count(self):
        names = self.tok.feature_names
        assert len(names) == 8 * len(BANDS)
