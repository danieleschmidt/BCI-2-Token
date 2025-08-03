"""
BCI-2-Token: Brain-Computer Interface to Token Translator

A Python framework for converting EEG/ECoG brain signals directly into 
LLM-compatible token logits with privacy-preserving differential privacy.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"

from .decoder import BrainDecoder
from .llm_interface import LLMInterface
from .preprocessing import SignalPreprocessor
from .streaming import StreamingDecoder
from .devices import create_device, DeviceManager
from .privacy import PrivacyEngine
from .training import BrainDecoderTrainer

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