"""
EEGTokenizer: Raw EEG (channels × timesteps) → token logit vector

Pipeline:
  1. Band-pass spectral feature extraction (delta/theta/alpha/beta/gamma)
  2. Per-band power via Welch's method (numpy FFT)
  3. Learned linear projection → token logit vector (via PyTorch nn.Linear)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpower(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Compute band power using Welch's periodogram method (numpy FFT)."""
    n = len(signal)
    # Simple periodogram via FFT
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(signal)) ** 2) / n
    # Integrate PSD within band
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(psd[mask]))


def extract_spectral_features(
    eeg: np.ndarray, fs: float = 256.0
) -> np.ndarray:
    """
    Extract spectral band power features from raw EEG.

    Args:
        eeg: shape (n_channels, n_timesteps) — raw EEG in microvolts
        fs:  sampling frequency in Hz

    Returns:
        features: shape (n_channels * n_bands,) — flattened band powers
    """
    if eeg.ndim != 2:
        raise ValueError(f"EEG must be 2D (channels × timesteps), got shape {eeg.shape}")

    n_channels = eeg.shape[0]
    n_bands = len(BANDS)
    features = np.zeros((n_channels, n_bands), dtype=np.float32)

    for ch_idx in range(n_channels):
        ch_signal = eeg[ch_idx]
        # Remove DC offset
        ch_signal = ch_signal - ch_signal.mean()
        for band_idx, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
            features[ch_idx, band_idx] = bandpower(ch_signal, fs, fmin, fmax)

    # Log-transform for numerical stability (band powers span many orders of magnitude)
    features = np.log1p(features)

    return features.flatten()


class EEGProjectionNet(nn.Module):
    """Two-layer MLP: spectral features → token logits."""

    def __init__(self, input_dim: int, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGTokenizer:
    """
    Converts raw EEG signals to token logit vectors.

    Usage:
        tok = EEGTokenizer(n_channels=32, vocab_size=50257, fs=256.0)
        eeg = np.random.randn(32, 512)   # 32 channels, 2 seconds @ 256 Hz
        logits = tok.encode(eeg)          # shape (vocab_size,)
    """

    def __init__(
        self,
        n_channels: int = 32,
        vocab_size: int = 50257,       # GPT-2 vocabulary by default
        fs: float = 256.0,
        hidden_dim: int = 256,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.n_channels = n_channels
        self.vocab_size = vocab_size
        self.fs = fs
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_dim = n_channels * len(BANDS)

        self.model = EEGProjectionNet(
            input_dim=self.input_dim,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        ).to(self.device)

        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.eval()

    def encode(self, eeg: np.ndarray) -> np.ndarray:
        """
        Encode a single EEG trial into token logits.

        Args:
            eeg: shape (n_channels, n_timesteps)

        Returns:
            logits: shape (vocab_size,) — raw (un-softmaxed) token scores
        """
        features = extract_spectral_features(eeg, fs=self.fs)  # (input_dim,)
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)  # (1, vocab_size)
        return logits.squeeze(0).cpu().numpy()

    def encode_batch(self, eegs: np.ndarray) -> np.ndarray:
        """
        Encode a batch of EEG trials.

        Args:
            eegs: shape (batch, n_channels, n_timesteps)

        Returns:
            logits: shape (batch, vocab_size)
        """
        features = np.stack([
            extract_spectral_features(eegs[i], fs=self.fs)
            for i in range(eegs.shape[0])
        ], axis=0)
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)  # (batch, vocab_size)
        return logits.cpu().numpy()

    @property
    def feature_names(self):
        """Returns feature names in the order they appear in the feature vector."""
        names = []
        for ch in range(self.n_channels):
            for band in BANDS:
                names.append(f"ch{ch}_{band}")
        return names
