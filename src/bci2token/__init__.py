"""
BCI-2-Token: Brain-Computer Interface to LLM Token Translation

A comprehensive framework for converting EEG/ECoG brain signals directly into
token logits compatible with any autoregressive language model, featuring
privacy-preserving differential privacy and state-of-the-art decoding accuracy.
"""

from .core.decoder import BrainDecoder
from .core.llm_interface import LLMInterface
from .streaming.realtime import StreamingDecoder
from .privacy.differential_privacy import PrivacyEngine
from .training.trainer import BrainDecoderTrainer
from .devices.base import Device
from .preprocessing.signal_processor import SignalProcessor

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

__all__ = [
    "BrainDecoder",
    "LLMInterface", 
    "StreamingDecoder",
    "PrivacyEngine",
    "BrainDecoderTrainer",
    "Device",
    "SignalProcessor",
]