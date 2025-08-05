"""
BCI-2-Token: Brain-Computer Interface to Token Translator

A Python framework for converting EEG/ECoG brain signals directly into 
LLM-compatible token logits with privacy-preserving differential privacy.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"

# Graceful imports with fallbacks for missing dependencies
try:
    from .decoder import BrainDecoder
    _HAS_DECODER = True
except ImportError as e:
    print(f"Warning: BrainDecoder not available: {e}")
    BrainDecoder = None
    _HAS_DECODER = False

try:
    from .llm_interface import LLMInterface
    _HAS_LLM = True
except ImportError as e:
    print(f"Warning: LLMInterface not available: {e}")
    LLMInterface = None
    _HAS_LLM = False

try:
    from .preprocessing import SignalPreprocessor
    _HAS_PREPROCESSING = True
except ImportError as e:
    print(f"Warning: SignalPreprocessor not available: {e}")
    SignalPreprocessor = None
    _HAS_PREPROCESSING = False

try:
    from .streaming import StreamingDecoder
    _HAS_STREAMING = True
except ImportError as e:
    print(f"Warning: StreamingDecoder not available: {e}")
    StreamingDecoder = None
    _HAS_STREAMING = False

try:
    from .devices import create_device, DeviceManager
    _HAS_DEVICES = True
except ImportError as e:
    print(f"Warning: Device support not available: {e}")
    create_device = None
    DeviceManager = None
    _HAS_DEVICES = False

try:
    from .privacy import PrivacyEngine
    _HAS_PRIVACY = True
except ImportError as e:
    print(f"Warning: PrivacyEngine not available: {e}")
    PrivacyEngine = None
    _HAS_PRIVACY = False

try:
    from .training import BrainDecoderTrainer
    _HAS_TRAINING = True
except ImportError as e:
    print(f"Warning: BrainDecoderTrainer not available: {e}")
    BrainDecoderTrainer = None
    _HAS_TRAINING = False

__all__ = [
    "BrainDecoder",
    "LLMInterface", 
    "SignalPreprocessor",
    "StreamingDecoder",
    "create_device",
    "DeviceManager", 
    "PrivacyEngine",
    "BrainDecoderTrainer"
]

# Export availability flags
__availability__ = {
    "decoder": _HAS_DECODER,
    "llm": _HAS_LLM,
    "preprocessing": _HAS_PREPROCESSING,
    "streaming": _HAS_STREAMING,
    "devices": _HAS_DEVICES,
    "privacy": _HAS_PRIVACY,
    "training": _HAS_TRAINING
}