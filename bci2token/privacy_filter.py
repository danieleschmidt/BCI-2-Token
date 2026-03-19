"""
PrivacyFilter: (ε, δ)-Differential Privacy for EEG feature vectors

Protects individual EEG feature representations via calibrated Gaussian noise
injection.  This prevents an adversary who sees the token logits from recovering
the original brain signal.

Theory (Gaussian Mechanism):
  Given sensitivity Δf (L2-norm of feature change between any two individuals),
  add noise N(0, σ²I) where:

      σ = Δf * √(2 ln(1.25/δ)) / ε

  This achieves (ε, δ)-DP.  See Dwork & Roth §A.1.

Usage:
    pf = PrivacyFilter(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    noisy_features = pf.apply(features)   # numpy array in, numpy array out
    noisy_logits   = pf.apply(logits)     # works on logits too
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class PrivacyBudget:
    """Tracks cumulative privacy spend under sequential composition."""
    epsilon: float
    delta: float
    _spent_epsilon: float = 0.0
    _spent_delta: float = 0.0
    _num_queries: int = 0

    def spend(self, epsilon: float, delta: float):
        """Record a query under basic composition (ε additively composes)."""
        self._spent_epsilon += epsilon
        self._spent_delta += self._num_queries * delta + delta  # basic composition
        self._num_queries += 1

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.epsilon - self._spent_epsilon)

    @property
    def is_exhausted(self) -> bool:
        return self._spent_epsilon >= self.epsilon or self._spent_delta >= self.delta

    def __repr__(self):
        return (
            f"PrivacyBudget(ε={self.epsilon}, δ={self.delta}, "
            f"spent_ε={self._spent_epsilon:.4f}, queries={self._num_queries})"
        )


class PrivacyFilter:
    """
    Applies calibrated Gaussian noise to EEG feature vectors for (ε, δ)-DP.

    Args:
        epsilon:     Privacy parameter ε > 0 (smaller = stronger privacy).
        delta:       Privacy parameter δ (probability of exceeding ε-guarantee).
                     Typically 1e-5 for neuroscience datasets.
        sensitivity: L2 sensitivity of the feature function Δf.
                     Defaults to 1.0; set to the max L2 norm difference
                     expected between any two individuals' feature vectors.
        clip_norm:   If set, clip input features to this L2 norm before noising
                     (enforces the sensitivity bound empirically).
        track_budget: Whether to track privacy budget spent per call.
        rng_seed:    Optional seed for reproducible noise.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        clip_norm: Optional[float] = None,
        track_budget: bool = True,
        rng_seed: Optional[int] = None,
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0,1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        self.track_budget = track_budget
        self._rng = np.random.default_rng(rng_seed)

        # Compute noise std: σ = Δf * sqrt(2 * ln(1.25/δ)) / ε
        self.sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

        if track_budget:
            self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)

    def apply(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian mechanism to a feature vector.

        Args:
            features: np.ndarray of any shape — noise is added element-wise.

        Returns:
            Noisy version of features (same shape and dtype).
        """
        features = np.asarray(features, dtype=np.float64)
        original_shape = features.shape
        flat = features.flatten()

        # Optional L2 clipping to enforce sensitivity bound
        if self.clip_norm is not None:
            norm = np.linalg.norm(flat)
            if norm > self.clip_norm:
                flat = flat * (self.clip_norm / norm)

        noise = self._rng.normal(loc=0.0, scale=self.sigma, size=flat.shape)
        noisy = flat + noise

        if self.track_budget:
            self.budget.spend(self.epsilon, self.delta)

        return noisy.reshape(original_shape).astype(features.dtype)

    def apply_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """
        Apply to a batch of feature vectors (first dimension = batch).

        Args:
            features_batch: shape (batch, ...) — each slice is a sample.

        Returns:
            Noisy batch, same shape.
        """
        return np.stack([self.apply(features_batch[i]) for i in range(features_batch.shape[0])])

    @property
    def noise_std(self) -> float:
        """Gaussian noise standard deviation used."""
        return self.sigma

    def __repr__(self):
        return (
            f"PrivacyFilter(ε={self.epsilon}, δ={self.delta}, "
            f"Δf={self.sensitivity}, σ={self.sigma:.4f})"
        )
