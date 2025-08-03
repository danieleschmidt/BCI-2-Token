"""
Differential privacy protection for brain signals.

Implements noise injection mechanisms to protect neural data privacy
while preserving decoding accuracy.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy protection."""
    
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Failure probability
    mechanism: str = 'gaussian'  # 'gaussian' or 'laplace'
    clip_norm: float = 1.0  # Gradient/signal clipping norm
    noise_multiplier: Optional[float] = None  # Auto-calculated if None
    adaptive_clipping: bool = True  # Use adaptive clipping
    per_sample_gradients: bool = False  # For training DP
    

class PrivacyEngine:
    """
    Differential privacy engine for protecting brain signals.
    
    Implements various DP mechanisms including Gaussian and Laplacian noise
    injection with adaptive clipping and privacy accounting.
    """
    
    def __init__(self, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 signal_channels: int = 64,
                 sampling_rate: int = 256,
                 mechanism: str = 'gaussian'):
        """
        Initialize privacy engine.
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Failure probability for (ε,δ)-DP
            signal_channels: Number of signal channels
            sampling_rate: Signal sampling rate
            mechanism: Noise mechanism ('gaussian' or 'laplace')
        """
        self.config = PrivacyConfig(
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism
        )
        
        self.signal_channels = signal_channels
        self.sampling_rate = sampling_rate
        
        # Calculate noise parameters
        self._calculate_noise_parameters()
        
        # Privacy accounting
        self.privacy_spent = 0.0
        self.noise_added = 0.0
        
    def _calculate_noise_parameters(self):
        """Calculate noise parameters based on privacy requirements."""
        if self.config.mechanism == 'gaussian':
            # For Gaussian mechanism: σ = √(2 ln(1.25/δ)) * Δf / ε
            # Where Δf is the sensitivity (L2 norm)
            sensitivity = self.config.clip_norm
            
            if self.config.noise_multiplier is None:
                # Calculate noise multiplier for (ε,δ)-DP
                self.noise_multiplier = np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.epsilon
            else:
                self.noise_multiplier = self.config.noise_multiplier
                
            self.noise_scale = sensitivity * self.noise_multiplier
            
        elif self.config.mechanism == 'laplace':
            # For Laplace mechanism: b = Δf / ε
            sensitivity = self.config.clip_norm
            self.noise_scale = sensitivity / self.config.epsilon
            self.noise_multiplier = self.noise_scale / sensitivity
            
        else:
            raise ValueError(f"Unknown mechanism: {self.config.mechanism}")
            
    def add_noise(self, data: np.ndarray, clip_data: bool = True) -> np.ndarray:
        """
        Add differential privacy noise to data.
        
        Args:
            data: Input data array
            clip_data: Whether to clip data before adding noise
            
        Returns:
            Noisy data with same shape as input
        """
        data = data.copy()
        
        # Clip data if requested
        if clip_data:
            data = self._clip_data(data)
            
        # Generate and add noise
        if self.config.mechanism == 'gaussian':
            noise = np.random.normal(0, self.noise_scale, data.shape)
        elif self.config.mechanism == 'laplace':
            noise = np.random.laplace(0, self.noise_scale, data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {self.config.mechanism}")
            
        noisy_data = data + noise
        
        # Update privacy accounting
        self.noise_added += np.mean(np.abs(noise))
        
        return noisy_data
        
    def _clip_data(self, data: np.ndarray) -> np.ndarray:
        """
        Clip data to bounded L2 norm.
        
        Args:
            data: Input data
            
        Returns:
            Clipped data
        """
        if data.ndim == 1:
            # Single vector
            norm = np.linalg.norm(data)
            if norm > self.config.clip_norm:
                data = data * (self.config.clip_norm / norm)
        elif data.ndim == 2:
            # Matrix: clip each row/column based on interpretation
            if data.shape[0] == self.signal_channels:
                # Channels × time: clip each channel
                for i in range(data.shape[0]):
                    norm = np.linalg.norm(data[i])
                    if norm > self.config.clip_norm:
                        data[i] = data[i] * (self.config.clip_norm / norm)
            else:
                # Treat as batch of vectors
                for i in range(data.shape[0]):
                    norm = np.linalg.norm(data[i])
                    if norm > self.config.clip_norm:
                        data[i] = data[i] * (self.config.clip_norm / norm)
        elif data.ndim == 3:
            # 3D data: epochs × channels × time
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    norm = np.linalg.norm(data[i, j])
                    if norm > self.config.clip_norm:
                        data[i, j] = data[i, j] * (self.config.clip_norm / norm)
        else:
            # General case: clip each sample in batch
            batch_size = data.shape[0]
            data_flat = data.reshape(batch_size, -1)
            
            for i in range(batch_size):
                norm = np.linalg.norm(data_flat[i])
                if norm > self.config.clip_norm:
                    data_flat[i] = data_flat[i] * (self.config.clip_norm / norm)
                    
            data = data_flat.reshape(data.shape)
            
        return data
        
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add DP noise to model gradients during training.
        
        Args:
            gradients: Dictionary of parameter gradients
            
        Returns:
            Noisy gradients
        """
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            if grad is None:
                noisy_gradients[name] = None
                continue
                
            # Clip gradient
            grad_norm = torch.norm(grad)
            if grad_norm > self.config.clip_norm:
                grad = grad * (self.config.clip_norm / grad_norm)
                
            # Add noise
            if self.config.mechanism == 'gaussian':
                noise = torch.normal(0, self.noise_scale, grad.shape)
            elif self.config.mechanism == 'laplace':
                # PyTorch doesn't have Laplace distribution, use manual implementation
                uniform = torch.rand(grad.shape) - 0.5
                noise = -self.noise_scale * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
            else:
                raise ValueError(f"Unknown mechanism: {self.config.mechanism}")
                
            noisy_gradients[name] = grad + noise
            
        return noisy_gradients
        
    def calculate_privacy_loss(self, num_samples: int, num_epochs: int) -> Tuple[float, float]:
        """
        Calculate total privacy loss for training.
        
        Args:
            num_samples: Number of training samples
            num_epochs: Number of training epochs
            
        Returns:
            Tuple of (epsilon, delta) privacy loss
        """
        if self.config.mechanism == 'gaussian':
            # RDP accounting for Gaussian mechanism
            # This is a simplified calculation - use opacus or similar for exact accounting
            q = 1.0 / num_samples  # Sampling probability (assuming full batch)
            steps = num_epochs * num_samples
            
            # Approximate RDP calculation
            alpha = 2.0  # RDP order
            rdp = q * steps * (alpha * self.noise_multiplier**2) / 2
            
            # Convert RDP to (ε,δ)-DP
            epsilon = rdp + np.log(1/self.config.delta) / (alpha - 1)
            delta = self.config.delta
            
        else:  # Laplace mechanism
            # For Laplace, epsilon adds up linearly
            epsilon = self.config.epsilon * num_epochs
            delta = 0.0  # Pure DP
            
        return epsilon, delta
        
    def get_noise_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics about noise impact on data.
        
        Args:
            data: Original data before noise
            
        Returns:
            Dictionary with noise statistics
        """
        # Generate noise with same parameters
        if self.config.mechanism == 'gaussian':
            noise = np.random.normal(0, self.noise_scale, data.shape)
        else:
            noise = np.random.laplace(0, self.noise_scale, data.shape)
            
        # Calculate SNR
        signal_power = np.mean(data**2)
        noise_power = np.mean(noise**2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Calculate relative noise
        data_std = np.std(data)
        noise_std = np.std(noise)
        relative_noise = noise_std / data_std if data_std > 0 else float('inf')
        
        return {
            'snr_db': snr_db,
            'signal_std': data_std,
            'noise_std': noise_std,
            'relative_noise': relative_noise,
            'noise_scale': self.noise_scale,
            'noise_multiplier': self.noise_multiplier
        }
        
    def generate_privacy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive privacy report.
        
        Returns:
            Dictionary with privacy parameters and guarantees
        """
        return {
            'privacy_parameters': {
                'epsilon': self.config.epsilon,
                'delta': self.config.delta,
                'mechanism': self.config.mechanism,
                'clip_norm': self.config.clip_norm,
                'noise_multiplier': self.noise_multiplier,
                'noise_scale': self.noise_scale
            },
            'privacy_accounting': {
                'privacy_spent': self.privacy_spent,
                'noise_added': self.noise_added
            },
            'signal_parameters': {
                'channels': self.signal_channels,
                'sampling_rate': self.sampling_rate
            },
            'privacy_interpretation': {
                'epsilon_meaning': "Lower epsilon = stronger privacy (ε=0 is perfect privacy)",
                'delta_meaning': "Probability of privacy failure (should be << 1/n_individuals)",
                'recommended_epsilon': "1.0 for moderate privacy, 0.1 for strong privacy",
                'recommended_delta': "1e-5 for typical applications"
            }
        }
        
    def validate_privacy_parameters(self) -> Dict[str, Any]:
        """
        Validate privacy parameters and provide recommendations.
        
        Returns:
            Validation results and recommendations
        """
        issues = []
        recommendations = []
        
        # Check epsilon
        if self.config.epsilon > 10:
            issues.append("Epsilon is very large (weak privacy)")
            recommendations.append("Consider reducing epsilon to < 1.0")
        elif self.config.epsilon < 0.01:
            issues.append("Epsilon is very small (may hurt utility significantly)")
            recommendations.append("Consider increasing epsilon to 0.1-1.0")
            
        # Check delta
        if self.config.delta > 1e-3:
            issues.append("Delta is large (significant privacy failure probability)")
            recommendations.append("Consider reducing delta to < 1e-5")
            
        # Check noise scale
        if self.noise_scale > 10:
            issues.append("Noise scale is very large (may destroy signal quality)")
            recommendations.append("Consider increasing epsilon or reducing clip_norm")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'privacy_level': self._classify_privacy_level()
        }
        
    def _classify_privacy_level(self) -> str:
        """Classify privacy level based on epsilon."""
        if self.config.epsilon < 0.1:
            return "Very Strong"
        elif self.config.epsilon < 1.0:
            return "Strong"
        elif self.config.epsilon < 5.0:
            return "Moderate"
        elif self.config.epsilon < 10.0:
            return "Weak"
        else:
            return "Very Weak"