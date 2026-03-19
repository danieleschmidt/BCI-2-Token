"""
SyntheticEEGGenerator: Realistic synthetic EEG for testing the BCI-2-Token pipeline.

Different mental states produce distinct spectral profiles:

  ┌──────────────────────┬─────────────────────────────────────────────────┐
  │ Mental State         │ Dominant EEG Signature                          │
  ├──────────────────────┼─────────────────────────────────────────────────┤
  │ rest                 │ Strong alpha (8-13 Hz), low beta                │
  │ motor_imagery_left   │ Alpha/beta ERD (event-related desync) left hem. │
  │ motor_imagery_right  │ Alpha/beta ERD right hemisphere                 │
  │ mental_arithmetic    │ Elevated theta (4-8 Hz) frontal, suppressed alp │
  │ visual_attention     │ Elevated gamma (30-45 Hz) occipital             │
  └──────────────────────┴─────────────────────────────────────────────────┘

Signals are generated as a sum of sinusoids at band center frequencies
with amplitudes drawn from Gaussian distributions parameterized per state,
plus 1/f (pink) background noise.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# EEG band center frequencies (Hz)
BAND_CENTERS = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.5,
    "beta":  20.0,
    "gamma": 37.5,
}

# Per-state band amplitude profiles (mean μV, std μV)
# Format: {state: {band: (mean_amplitude, std_amplitude)}}
STATE_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "rest": {
        "delta": (1.0, 0.3),
        "theta": (1.5, 0.4),
        "alpha": (8.0, 1.5),   # dominant alpha at rest
        "beta":  (2.0, 0.5),
        "gamma": (0.5, 0.2),
    },
    "motor_imagery_left": {
        "delta": (1.2, 0.3),
        "theta": (2.0, 0.5),
        "alpha": (3.5, 0.8),   # ERD: alpha suppressed during imagery
        "beta":  (1.5, 0.4),   # beta also suppressed
        "gamma": (1.5, 0.4),   # slight gamma increase
    },
    "motor_imagery_right": {
        "delta": (1.2, 0.3),
        "theta": (2.0, 0.5),
        "alpha": (3.0, 0.8),
        "beta":  (1.2, 0.4),
        "gamma": (1.8, 0.4),
    },
    "mental_arithmetic": {
        "delta": (0.8, 0.3),
        "theta": (5.0, 1.0),   # elevated frontal theta during arithmetic
        "alpha": (2.5, 0.6),   # alpha suppressed
        "beta":  (3.5, 0.7),   # beta elevated (cognitive load)
        "gamma": (2.0, 0.5),
    },
    "visual_attention": {
        "delta": (0.8, 0.2),
        "theta": (1.8, 0.4),
        "alpha": (4.0, 0.8),   # alpha moderate
        "beta":  (2.5, 0.5),
        "gamma": (6.0, 1.2),   # elevated occipital gamma during visual attention
    },
}

MENTAL_STATES = list(STATE_PROFILES.keys())


@dataclass
class EEGTrial:
    """A single synthetic EEG trial."""
    data: np.ndarray          # shape (n_channels, n_timesteps) in μV
    state: str                # mental state label
    fs: float                 # sampling frequency in Hz
    duration_s: float         # trial duration in seconds

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_timesteps(self) -> int:
        return self.data.shape[1]


class SyntheticEEGGenerator:
    """
    Generates realistic synthetic EEG trials for BCI pipeline testing.

    Each mental state has a distinct spectral fingerprint based on empirical
    EEG literature for motor imagery, mental workload, and visual attention.

    Args:
        n_channels:   Number of EEG channels (default 32, mimics a 32-cap setup).
        fs:           Sampling frequency in Hz (default 256 Hz).
        duration_s:   Trial duration in seconds (default 2.0s).
        noise_level:  Scaling factor for pink background noise (default 0.5 μV).
        rng_seed:     Seed for reproducibility.
    """

    def __init__(
        self,
        n_channels: int = 32,
        fs: float = 256.0,
        duration_s: float = 2.0,
        noise_level: float = 0.5,
        rng_seed: Optional[int] = 42,
    ):
        self.n_channels = n_channels
        self.fs = fs
        self.duration_s = duration_s
        self.noise_level = noise_level
        self.rng = np.random.default_rng(rng_seed)

    def _pink_noise(self, n: int) -> np.ndarray:
        """Generate 1/f (pink) noise of length n."""
        white = self.rng.standard_normal(n)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-10  # avoid divide-by-zero
        pink_filter = 1.0 / np.sqrt(freqs)
        pink = np.fft.irfft(np.fft.rfft(white) * pink_filter, n=n)
        # Normalize to unit variance
        std = pink.std()
        if std > 0:
            pink /= std
        return pink

    def _generate_channel(
        self, state: str, channel_idx: int
    ) -> np.ndarray:
        """
        Generate one channel of EEG for a given mental state.

        Channels have slight spatial variation: frontal channels (first 1/4)
        get extra theta, occipital channels (last 1/4) get extra gamma.
        """
        n = int(self.fs * self.duration_s)
        t = np.linspace(0, self.duration_s, n, endpoint=False)
        signal = np.zeros(n)
        profile = STATE_PROFILES[state]

        is_frontal = channel_idx < self.n_channels // 4
        is_occipital = channel_idx >= 3 * self.n_channels // 4

        for band, center_freq in BAND_CENTERS.items():
            mean_amp, std_amp = profile[band]
            # Spatial modulation
            if band == "theta" and is_frontal:
                mean_amp *= 1.4
            if band == "gamma" and is_occipital:
                mean_amp *= 1.6
            if band == "alpha" and is_occipital:
                mean_amp *= 0.7  # occipital alpha is weaker

            amplitude = max(0.0, self.rng.normal(mean_amp, std_amp))
            phase = self.rng.uniform(0, 2 * np.pi)
            signal += amplitude * np.sin(2 * np.pi * center_freq * t + phase)

        # Add pink background noise
        signal += self.noise_level * self._pink_noise(n)

        return signal.astype(np.float32)

    def generate(self, state: str) -> EEGTrial:
        """
        Generate a single EEG trial for a given mental state.

        Args:
            state: One of MENTAL_STATES = ['rest', 'motor_imagery_left',
                   'motor_imagery_right', 'mental_arithmetic', 'visual_attention']

        Returns:
            EEGTrial with data shape (n_channels, n_timesteps)
        """
        if state not in STATE_PROFILES:
            raise ValueError(
                f"Unknown state '{state}'. Choose from: {list(STATE_PROFILES.keys())}"
            )
        data = np.stack([
            self._generate_channel(state, ch) for ch in range(self.n_channels)
        ], axis=0)
        return EEGTrial(
            data=data,
            state=state,
            fs=self.fs,
            duration_s=self.duration_s,
        )

    def generate_batch(self, state: str, n_trials: int) -> np.ndarray:
        """
        Generate multiple trials for a state.

        Returns:
            ndarray shape (n_trials, n_channels, n_timesteps)
        """
        return np.stack([self.generate(state).data for _ in range(n_trials)], axis=0)

    def generate_dataset(
        self, n_per_state: int = 10
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate a balanced dataset across all mental states.

        Returns:
            (X, y): X shape (n_total, n_channels, n_timesteps), y list of state labels
        """
        X, y = [], []
        for state in MENTAL_STATES:
            trials = self.generate_batch(state, n_per_state)
            X.append(trials)
            y.extend([state] * n_per_state)
        return np.concatenate(X, axis=0), y
