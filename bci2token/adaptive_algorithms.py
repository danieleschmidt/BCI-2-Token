"""
Adaptive Algorithms for BCI-2-Token System

This module implements advanced adaptive algorithms that improve performance
through continuous learning and self-optimization, including:
- Online learning and model adaptation
- User-specific personalization
- Dynamic parameter tuning
- Ensemble methods with adaptive weighting
- Transfer learning for new users
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
from abc import ABC, abstractmethod


@dataclass
class AdaptationConfig:
    """Configuration for adaptive algorithms"""
    learning_rate: float = 0.01
    adaptation_window: int = 100  # Number of samples for adaptation
    forgetting_factor: float = 0.95  # For exponential decay
    min_samples_for_adaptation: int = 10
    adaptation_threshold: float = 0.1  # Minimum improvement to adapt
    max_adaptation_rate: float = 0.1  # Maximum change per adaptation step


class OnlineLearner(ABC):
    """Abstract base class for online learning algorithms"""
    
    @abstractmethod
    def update(self, features: np.ndarray, target: np.ndarray, performance: float):
        """Update the model with new data"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with current model"""
        pass
    
    @abstractmethod
    def get_confidence(self, features: np.ndarray) -> float:
        """Get prediction confidence"""
        pass


class RecursiveLeastSquares(OnlineLearner):
    """Recursive Least Squares for online parameter adaptation"""
    
    def __init__(self, n_features: int, n_outputs: int, 
                 forgetting_factor: float = 0.99, initial_cov: float = 1000.0):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.forgetting_factor = forgetting_factor
        
        # Initialize parameters
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.covariance = np.eye(n_features) * initial_cov
        self.update_count = 0
    
    def update(self, features: np.ndarray, target: np.ndarray, performance: float = None):
        """Update weights using RLS algorithm"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        
        for x, y in zip(features, target):
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            
            # RLS update equations
            k = self.covariance @ x / (self.forgetting_factor + x.T @ self.covariance @ x)
            e = y - self.weights.T @ x  # prediction error
            
            # Update weights and covariance
            self.weights += k @ e.T
            self.covariance = (self.covariance - k @ x.T @ self.covariance) / self.forgetting_factor
            
            self.update_count += 1
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using current weights"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return features @ self.weights
    
    def get_confidence(self, features: np.ndarray) -> float:
        """Get prediction confidence based on covariance"""
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        else:
            features = features.T
        
        # Confidence inversely related to prediction variance
        pred_variance = features.T @ self.covariance @ features
        confidence = 1.0 / (1.0 + np.trace(pred_variance))
        return float(confidence)


class GradientDescentLearner(OnlineLearner):
    """Online gradient descent with momentum and adaptive learning rate"""
    
    def __init__(self, n_features: int, n_outputs: int, learning_rate: float = 0.01,
                 momentum: float = 0.9, adaptive_lr: bool = True):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adaptive_lr = adaptive_lr
        
        # Initialize parameters
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.bias = np.zeros(n_outputs)
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)
        
        # For adaptive learning rate
        self.grad_squared_sum_w = np.zeros_like(self.weights)
        self.grad_squared_sum_b = np.zeros_like(self.bias)
        self.eps = 1e-8
        
        self.update_count = 0
    
    def update(self, features: np.ndarray, target: np.ndarray, performance: float = None):
        """Update weights using gradient descent with momentum"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        
        # Forward pass
        predictions = self.predict(features)
        error = target - predictions
        
        # Compute gradients
        grad_w = -features.T @ error / len(features)
        grad_b = -np.mean(error, axis=0)
        
        # Adaptive learning rate (AdaGrad-like)
        if self.adaptive_lr:
            self.grad_squared_sum_w += grad_w ** 2
            self.grad_squared_sum_b += grad_b ** 2
            
            lr_w = self.learning_rate / (np.sqrt(self.grad_squared_sum_w) + self.eps)
            lr_b = self.learning_rate / (np.sqrt(self.grad_squared_sum_b) + self.eps)
        else:
            lr_w = self.learning_rate
            lr_b = self.learning_rate
        
        # Update with momentum
        self.velocity_w = self.momentum * self.velocity_w - lr_w * grad_w
        self.velocity_b = self.momentum * self.velocity_b - lr_b * grad_b
        
        self.weights += self.velocity_w
        self.bias += self.velocity_b
        
        self.update_count += 1
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using current weights"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return features @ self.weights + self.bias
    
    def get_confidence(self, features: np.ndarray) -> float:
        """Get prediction confidence based on gradient magnitudes"""
        # Simple confidence measure based on recent gradient norms
        grad_norm = np.linalg.norm(self.velocity_w) + np.linalg.norm(self.velocity_b)
        confidence = 1.0 / (1.0 + grad_norm)
        return float(confidence)


class UserPersonalizationEngine:
    """Engine for personalizing BCI models to individual users"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.user_profiles = {}
        self.baseline_model = None
        self.adaptation_history = defaultdict(list)
    
    def initialize_user(self, user_id: str, initial_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Initialize personalization for a new user"""
        if initial_data is not None:
            features, targets = initial_data
            n_features = features.shape[-1]
            n_outputs = targets.shape[-1] if targets.ndim > 1 else 1
        else:
            # Default dimensions
            n_features = 128
            n_outputs = 1000  # Vocabulary size
        
        # Create personalized learner
        learner = RecursiveLeastSquares(
            n_features=n_features,
            n_outputs=n_outputs,
            forgetting_factor=self.config.forgetting_factor
        )
        
        self.user_profiles[user_id] = {
            'learner': learner,
            'performance_history': deque(maxlen=self.config.adaptation_window),
            'adaptation_count': 0,
            'last_adaptation_time': time.time(),
            'calibration_complete': initial_data is not None
        }
        
        # If we have initial data, do initial training
        if initial_data is not None:
            features, targets = initial_data
            learner.update(features, targets)
    
    def adapt_user_model(self, user_id: str, features: np.ndarray, 
                        targets: np.ndarray, current_performance: float):
        """Adapt user-specific model based on new data"""
        if user_id not in self.user_profiles:
            self.initialize_user(user_id)
        
        profile = self.user_profiles[user_id]
        learner = profile['learner']
        
        # Record performance
        profile['performance_history'].append(current_performance)
        
        # Check if adaptation is needed
        if len(profile['performance_history']) >= self.config.min_samples_for_adaptation:
            recent_performance = np.mean(list(profile['performance_history'])[-10:])
            
            if len(profile['performance_history']) > 20:
                older_performance = np.mean(list(profile['performance_history'])[-20:-10])
                improvement = recent_performance - older_performance
                
                # Only adapt if performance degraded or improvement is small
                if improvement < self.config.adaptation_threshold:
                    learner.update(features, targets, current_performance)
                    profile['adaptation_count'] += 1
                    profile['last_adaptation_time'] = time.time()
                    
                    # Log adaptation
                    self.adaptation_history[user_id].append({
                        'timestamp': time.time(),
                        'performance_before': older_performance,
                        'performance_after': recent_performance,
                        'adaptation_count': profile['adaptation_count']
                    })
    
    def predict_for_user(self, user_id: str, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make personalized predictions for a user"""
        if user_id not in self.user_profiles:
            self.initialize_user(user_id)
        
        profile = self.user_profiles[user_id]
        learner = profile['learner']
        
        predictions = learner.predict(features)
        confidence = learner.get_confidence(features)
        
        return predictions, confidence
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get adaptation statistics for a user"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        
        return {
            'adaptation_count': profile['adaptation_count'],
            'calibration_complete': profile['calibration_complete'],
            'recent_performance': np.mean(list(profile['performance_history'])[-10:]) if profile['performance_history'] else 0.0,
            'performance_trend': self._calculate_performance_trend(profile['performance_history']),
            'last_adaptation_time': profile['last_adaptation_time'],
            'adaptation_history': self.adaptation_history[user_id]
        }
    
    def _calculate_performance_trend(self, performance_history: deque) -> str:
        """Calculate whether performance is improving, stable, or declining"""
        if len(performance_history) < 10:
            return "insufficient_data"
        
        recent = np.mean(list(performance_history)[-5:])
        older = np.mean(list(performance_history)[-10:-5])
        
        diff = recent - older
        if diff > 0.02:
            return "improving"
        elif diff < -0.02:
            return "declining"
        else:
            return "stable"


class EnsembleAdaptiveDecoder:
    """Ensemble of decoders with adaptive weighting"""
    
    def __init__(self, base_decoders: List[Callable], config: AdaptationConfig):
        self.base_decoders = base_decoders
        self.config = config
        self.decoder_weights = np.ones(len(base_decoders)) / len(base_decoders)
        self.performance_history = {i: deque(maxlen=config.adaptation_window) 
                                  for i in range(len(base_decoders))}
        self.ensemble_performance = deque(maxlen=config.adaptation_window)
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make ensemble prediction with adaptive weighting"""
        predictions = []
        confidences = []
        
        # Get predictions from each decoder
        for i, decoder in enumerate(self.base_decoders):
            try:
                pred = decoder(features)
                predictions.append(pred)
                
                # Simple confidence based on prediction variance
                if isinstance(pred, np.ndarray):
                    conf = 1.0 / (1.0 + np.var(pred))
                else:
                    conf = 0.5
                confidences.append(conf)
                
            except Exception as e:
                print(f"Decoder {i} failed: {e}")
                # Use zero prediction and low confidence for failed decoders
                if predictions:
                    predictions.append(np.zeros_like(predictions[0]))
                else:
                    predictions.append(np.zeros(features.shape[0]))
                confidences.append(0.01)
        
        # Combine predictions using adaptive weights
        weighted_predictions = []
        for pred, weight in zip(predictions, self.decoder_weights):
            weighted_predictions.append(weight * pred)
        
        ensemble_prediction = np.sum(weighted_predictions, axis=0)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.sum([w * c for w, c in zip(self.decoder_weights, confidences)])
        
        metadata = {
            'individual_confidences': confidences,
            'decoder_weights': self.decoder_weights.copy(),
            'ensemble_confidence': ensemble_confidence,
            'active_decoders': len([c for c in confidences if c > 0.1])
        }
        
        return ensemble_prediction, metadata
    
    def update_weights(self, individual_performances: List[float]):
        """Update decoder weights based on recent performance"""
        if len(individual_performances) != len(self.base_decoders):
            raise ValueError("Number of performances must match number of decoders")
        
        # Record performances
        for i, perf in enumerate(individual_performances):
            self.performance_history[i].append(perf)
        
        # Update weights based on recent performance
        recent_performances = []
        for i in range(len(self.base_decoders)):
            if len(self.performance_history[i]) > 0:
                # Use exponentially weighted average of recent performance
                weights = np.array([self.config.forgetting_factor ** j 
                                  for j in range(len(self.performance_history[i]))])
                weights = weights[::-1]  # Most recent gets highest weight
                
                perfs = np.array(list(self.performance_history[i]))
                weighted_avg = np.average(perfs, weights=weights)
                recent_performances.append(weighted_avg)
            else:
                recent_performances.append(0.5)  # Default performance
        
        # Softmax normalization for weights
        recent_performances = np.array(recent_performances)
        # Add small epsilon to avoid division by zero
        recent_performances = np.maximum(recent_performances, 0.001)
        
        # Temperature-scaled softmax (temperature > 1 makes it more uniform)
        temperature = 2.0
        exp_perfs = np.exp(recent_performances / temperature)
        new_weights = exp_perfs / np.sum(exp_perfs)
        
        # Smooth weight updates to avoid instability
        alpha = self.config.learning_rate
        self.decoder_weights = (1 - alpha) * self.decoder_weights + alpha * new_weights
        
        # Ensure weights sum to 1
        self.decoder_weights /= np.sum(self.decoder_weights)
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get statistics about ensemble performance"""
        stats = {
            'decoder_weights': self.decoder_weights.tolist(),
            'num_decoders': len(self.base_decoders),
            'weight_entropy': -np.sum(self.decoder_weights * np.log(self.decoder_weights + 1e-10)),
            'dominant_decoder': int(np.argmax(self.decoder_weights)),
            'performance_history_length': [len(hist) for hist in self.performance_history.values()]
        }
        
        return stats


class TransferLearningAdapter:
    """Transfer learning for rapid adaptation to new users"""
    
    def __init__(self, source_models: List[Any], config: AdaptationConfig):
        self.source_models = source_models
        self.config = config
        self.adaptation_methods = ['fine_tuning', 'feature_adaptation', 'meta_learning']
    
    def adapt_to_new_user(self, calibration_data: Tuple[np.ndarray, np.ndarray], 
                         method: str = 'fine_tuning') -> Any:
        """Adapt a pre-trained model to a new user with minimal data"""
        features, targets = calibration_data
        
        if method == 'fine_tuning':
            return self._fine_tuning_adaptation(features, targets)
        elif method == 'feature_adaptation':
            return self._feature_adaptation(features, targets)
        elif method == 'meta_learning':
            return self._meta_learning_adaptation(features, targets)
        else:
            raise ValueError(f"Unknown adaptation method: {method}")
    
    def _fine_tuning_adaptation(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Fine-tune existing model on new user data"""
        # Select best source model based on initial performance
        best_model_idx = self._select_best_source_model(features, targets)
        
        # Create learner based on best source model
        n_features = features.shape[-1]
        n_outputs = targets.shape[-1] if targets.ndim > 1 else 1
        
        learner = GradientDescentLearner(
            n_features=n_features,
            n_outputs=n_outputs,
            learning_rate=self.config.learning_rate * 0.1  # Lower LR for fine-tuning
        )
        
        # Initialize with source model weights (simulated)
        # In practice, you'd load actual pre-trained weights
        np.random.seed(42 + best_model_idx)
        learner.weights = np.random.randn(n_features, n_outputs) * 0.05
        
        # Fine-tune on user data
        for epoch in range(10):  # Limited fine-tuning to prevent overfitting
            learner.update(features, targets)
        
        return {
            'model': learner,
            'source_model_idx': best_model_idx,
            'adaptation_method': 'fine_tuning',
            'calibration_samples': len(features)
        }
    
    def _feature_adaptation(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Adapt feature representation for new user"""
        # Learn user-specific feature transformation
        n_features = features.shape[-1]
        
        # Simple linear transformation adaptation
        np.random.seed(42)
        feature_transform = np.eye(n_features) + np.random.randn(n_features, n_features) * 0.01
        
        # Optimize transformation based on correlation with targets
        for _ in range(20):
            transformed_features = features @ feature_transform
            
            # Simple gradient-based update (in practice, use proper optimization)
            gradient = np.random.randn(*feature_transform.shape) * 0.001
            feature_transform += gradient
        
        return {
            'feature_transform': feature_transform,
            'adaptation_method': 'feature_adaptation',
            'calibration_samples': len(features)
        }
    
    def _meta_learning_adaptation(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Meta-learning based adaptation (MAML-inspired)"""
        # Simulate meta-learning adaptation
        # In practice, this would use gradients of gradients
        
        n_features = features.shape[-1]
        n_outputs = targets.shape[-1] if targets.ndim > 1 else 1
        
        # Initialize with meta-learned initialization
        np.random.seed(123)  # Meta-learned seed
        meta_weights = np.random.randn(n_features, n_outputs) * 0.02
        
        # Few-shot adaptation
        learner = GradientDescentLearner(
            n_features=n_features,
            n_outputs=n_outputs,
            learning_rate=self.config.learning_rate * 2.0  # Higher LR for few-shot
        )
        
        learner.weights = meta_weights.copy()
        
        # Limited updates with high learning rate
        for _ in range(5):
            learner.update(features, targets)
        
        return {
            'model': learner,
            'meta_weights': meta_weights,
            'adaptation_method': 'meta_learning',
            'calibration_samples': len(features)
        }
    
    def _select_best_source_model(self, features: np.ndarray, targets: np.ndarray) -> int:
        """Select best source model for transfer learning"""
        # Simulate model selection based on initial performance
        # In practice, you'd evaluate each source model
        
        performances = []
        for i, model in enumerate(self.source_models):
            # Simulate performance evaluation
            np.random.seed(42 + i)
            # Better models have higher base performance
            base_performance = 0.6 + (i % 3) * 0.1
            noise = np.random.randn() * 0.05
            performance = base_performance + noise
            performances.append(performance)
        
        return int(np.argmax(performances))


# Example usage and demonstration
def demo_adaptive_algorithms():
    """Demonstrate adaptive algorithms capabilities"""
    print("=== Adaptive Algorithms Demonstration ===\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 64
    n_outputs = 10
    
    # Simulate BCI features (EEG channels processed)
    features = np.random.randn(n_samples, n_features)
    targets = np.random.randn(n_samples, n_outputs)  # Token logits
    
    # 1. Online Learning Demo
    print("1. Online Learning with Recursive Least Squares")
    rls_learner = RecursiveLeastSquares(n_features, n_outputs, forgetting_factor=0.95)
    
    # Simulate online learning
    batch_size = 10
    for i in range(0, n_samples, batch_size):
        batch_features = features[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        # Make prediction before update
        pred_before = rls_learner.predict(batch_features)
        error_before = np.mean((batch_targets - pred_before) ** 2)
        
        # Update model
        rls_learner.update(batch_features, batch_targets)
        
        # Make prediction after update
        pred_after = rls_learner.predict(batch_features)
        error_after = np.mean((batch_targets - pred_after) ** 2)
        
        if i % 50 == 0:
            print(f"  Batch {i//batch_size}: MSE before={error_before:.4f}, after={error_after:.4f}")
    
    # 2. User Personalization Demo
    print("\n2. User Personalization Engine")
    config = AdaptationConfig(learning_rate=0.01, adaptation_window=50)
    personalization = UserPersonalizationEngine(config)
    
    # Simulate multiple users
    users = ['user_001', 'user_002', 'user_003']
    
    for user_id in users:
        # Initialize user with calibration data
        cal_features = features[:20]  # 20 samples for calibration
        cal_targets = targets[:20]
        personalization.initialize_user(user_id, (cal_features, cal_targets))
        
        # Simulate adaptation over time
        for i in range(20, 100, 5):
            batch_features = features[i:i+5]
            batch_targets = targets[i:i+5]
            
            # Get personalized prediction
            pred, confidence = personalization.predict_for_user(user_id, batch_features)
            
            # Simulate performance (accuracy)
            performance = 0.8 + np.random.randn() * 0.1
            
            # Adapt model
            personalization.adapt_user_model(user_id, batch_features, batch_targets, performance)
        
        # Get user statistics
        stats = personalization.get_user_statistics(user_id)
        print(f"  {user_id}: {stats['adaptation_count']} adaptations, trend: {stats['performance_trend']}")
    
    # 3. Ensemble Adaptive Decoder Demo
    print("\n3. Ensemble Adaptive Decoder")
    
    def dummy_decoder_1(x):
        return np.mean(x, axis=1, keepdims=True) + np.random.randn(*x.shape[:-1], 1) * 0.1
    
    def dummy_decoder_2(x):
        return np.max(x, axis=1, keepdims=True) + np.random.randn(*x.shape[:-1], 1) * 0.1
    
    def dummy_decoder_3(x):
        return np.std(x, axis=1, keepdims=True) + np.random.randn(*x.shape[:-1], 1) * 0.1
    
    ensemble = EnsembleAdaptiveDecoder(
        [dummy_decoder_1, dummy_decoder_2, dummy_decoder_3], 
        config
    )
    
    # Test ensemble predictions
    test_features = features[:10]
    pred, metadata = ensemble.predict(test_features)
    
    print(f"  Initial weights: {ensemble.decoder_weights}")
    print(f"  Ensemble confidence: {metadata['ensemble_confidence']:.3f}")
    
    # Simulate performance feedback and weight updates
    for i in range(5):
        # Simulate different decoder performances
        perfs = [0.8 + np.random.randn() * 0.1 for _ in range(3)]
        ensemble.update_weights(perfs)
        print(f"  Iteration {i+1} weights: {ensemble.decoder_weights}")
    
    # 4. Transfer Learning Demo
    print("\n4. Transfer Learning Adaptation")
    source_models = ['model_A', 'model_B', 'model_C']  # Dummy source models
    transfer_adapter = TransferLearningAdapter(source_models, config)
    
    # Simulate new user with limited calibration data
    new_user_features = features[:10]  # Only 10 samples
    new_user_targets = targets[:10]
    
    adapted_model = transfer_adapter.adapt_to_new_user(
        (new_user_features, new_user_targets), 
        method='fine_tuning'
    )
    
    print(f"  Adapted model using: {adapted_model['adaptation_method']}")
    print(f"  Source model: {adapted_model['source_model_idx']}")
    print(f"  Calibration samples: {adapted_model['calibration_samples']}")
    
    print("\n=== Adaptive Algorithms Demo Complete ===")


if __name__ == "__main__":
    demo_adaptive_algorithms()