"""
Pytest configuration and shared fixtures for BCI-2-Token testing
"""

import pytest
import numpy as np
import torch
from typing import Generator, Dict, Any, List, Tuple
import tempfile
import shutil
from pathlib import Path

from bci2token.core.decoder import BrainDecoder, DecoderConfig, DecoderType
from bci2token.agents.base_agent import AgentContext
from bci2token.preprocessing.signal_processor import SignalProcessor


@pytest.fixture
def sample_eeg_data() -> np.ndarray:
    """Generate sample EEG data for testing"""
    np.random.seed(42)  # Reproducible test data
    channels = 64
    time_points = 1000
    sampling_rate = 256
    
    # Generate realistic EEG-like signals
    time = np.linspace(0, time_points / sampling_rate, time_points)
    signals = np.zeros((channels, time_points))
    
    for ch in range(channels):
        # Add multiple frequency components
        alpha = 0.5 * np.sin(2 * np.pi * 10 * time)  # Alpha waves (10 Hz)
        beta = 0.3 * np.sin(2 * np.pi * 20 * time)   # Beta waves (20 Hz)
        gamma = 0.2 * np.sin(2 * np.pi * 40 * time)  # Gamma waves (40 Hz)
        noise = 0.1 * np.random.randn(time_points)    # Noise
        
        signals[ch] = alpha + beta + gamma + noise
    
    return signals


@pytest.fixture
def sample_ecog_data() -> np.ndarray:
    """Generate sample ECoG data for testing"""
    np.random.seed(123)
    channels = 128
    time_points = 2000  # Higher resolution for ECoG
    sampling_rate = 1000
    
    time = np.linspace(0, time_points / sampling_rate, time_points)
    signals = np.zeros((channels, time_points))
    
    for ch in range(channels):
        # ECoG typically has higher frequency content
        high_gamma = 0.4 * np.sin(2 * np.pi * 80 * time)
        broadband = 0.3 * np.random.randn(time_points)
        signals[ch] = high_gamma + broadband
    
    return signals


@pytest.fixture
def sample_tokens() -> List[int]:
    """Sample token sequence for testing"""
    return [15496, 995, 318, 257, 1332, 2646]  # "Hello world is a test message"


@pytest.fixture
def decoder_config() -> DecoderConfig:
    """Default decoder configuration for testing"""
    return DecoderConfig(
        signal_type="eeg",
        channels=64,
        sampling_rate=256,
        decoder_type=DecoderType.CTC_CONFORMER,
        vocab_size=50257,
        hidden_dim=256,  # Smaller for faster testing
        num_layers=2,    # Fewer layers for faster testing
        num_heads=4,
        dropout=0.1,
        device="cpu"     # Force CPU for consistent testing
    )


@pytest.fixture
def brain_decoder(decoder_config: DecoderConfig) -> BrainDecoder:
    """Initialize brain decoder for testing"""
    return BrainDecoder(decoder_config)


@pytest.fixture
def agent_context() -> AgentContext:
    """Sample agent context for testing"""
    return AgentContext(
        project_root="/tmp/test_project",
        current_branch="test-branch",
        requirements={"test_requirement": "value"},
        architecture={"test_component": "design"}
    )


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def training_data() -> List[Tuple[np.ndarray, List[int]]]:
    """Generate training data for testing"""
    np.random.seed(456)
    data = []
    
    for i in range(5):  # Small dataset for testing
        signals = np.random.randn(64, 500)
        tokens = [100 + i, 200 + i, 300 + i]  # Simple token patterns
        data.append((signals, tokens))
    
    return data


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Mock LLM response for testing"""
    return {
        "tokens": [15496, 995, 318, 257, 1332],
        "text": "Hello world is a test",
        "confidence": 0.85,
        "logits": torch.randn(5, 50257).tolist()
    }


@pytest.fixture(scope="session")
def performance_test_data() -> Dict[str, np.ndarray]:
    """Large datasets for performance testing"""
    return {
        "large_eeg": np.random.randn(128, 10000),
        "large_ecog": np.random.randn(256, 20000),
        "batch_signals": np.random.randn(32, 64, 1000)  # Batch of 32 signals
    }


@pytest.fixture
def security_test_inputs() -> Dict[str, Any]:
    """Inputs for security testing"""
    return {
        "malicious_signals": np.array([[1e10, -1e10] * 500] * 64),  # Extreme values
        "nan_signals": np.full((64, 1000), np.nan),                  # NaN values
        "inf_signals": np.full((64, 1000), np.inf),                  # Infinite values
        "zero_signals": np.zeros((64, 1000)),                        # All zeros
        "adversarial_tokens": [-1, 50257, 100000],                   # Out of bounds tokens
    }


@pytest.fixture
def privacy_test_config() -> DecoderConfig:
    """Configuration with privacy settings for testing"""
    return DecoderConfig(
        signal_type="eeg",
        channels=64,
        sampling_rate=256,
        decoder_type=DecoderType.CTC_CONFORMER,
        privacy_epsilon=1.0,  # Enable differential privacy
        hidden_dim=128,
        num_layers=2,
        device="cpu"
    )


# Markers for different test types
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def capture_logs(caplog):
    """Capture logs with specific configuration"""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def generate_brain_signals(
        signal_type: str, 
        channels: int, 
        duration_seconds: float,
        sampling_rate: int = 256
    ) -> np.ndarray:
        """Generate realistic brain signals for testing"""
        time_points = int(duration_seconds * sampling_rate)
        time = np.linspace(0, duration_seconds, time_points)
        signals = np.zeros((channels, time_points))
        
        if signal_type == "eeg":
            # EEG frequency bands
            delta = 0.3 * np.sin(2 * np.pi * 2 * time)   # 1-4 Hz
            theta = 0.4 * np.sin(2 * np.pi * 6 * time)   # 4-8 Hz
            alpha = 0.5 * np.sin(2 * np.pi * 10 * time)  # 8-13 Hz
            beta = 0.3 * np.sin(2 * np.pi * 20 * time)   # 13-30 Hz
            gamma = 0.2 * np.sin(2 * np.pi * 40 * time)  # 30-100 Hz
            
            for ch in range(channels):
                noise = 0.1 * np.random.randn(time_points)
                signals[ch] = delta + theta + alpha + beta + gamma + noise
                
        elif signal_type == "ecog":
            # ECoG has higher frequency content
            for ch in range(channels):
                high_gamma = 0.4 * np.sin(2 * np.pi * 80 * time)
                broadband = 0.3 * np.random.randn(time_points)
                signals[ch] = high_gamma + broadband
                
        return signals
    
    @staticmethod
    def generate_token_sequences(
        num_sequences: int, 
        min_length: int = 5, 
        max_length: int = 20,
        vocab_size: int = 50257
    ) -> List[List[int]]:
        """Generate token sequences for testing"""
        sequences = []
        for _ in range(num_sequences):
            length = np.random.randint(min_length, max_length + 1)
            sequence = np.random.randint(0, vocab_size, length).tolist()
            sequences.append(sequence)
        return sequences


@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator utility"""
    return TestDataGenerator()