"""
BCI-2-Token: EEG Brain Signals → Language Model Token Probabilities

A focused pipeline for converting raw EEG signals into LLM-compatible
token logit vectors, with differential privacy protection.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .eeg_tokenizer import EEGTokenizer
from .token_decoder import TokenDecoder
from .privacy_filter import PrivacyFilter
from .synthetic_eeg import SyntheticEEGGenerator

__all__ = [
    "EEGTokenizer",
    "TokenDecoder",
    "PrivacyFilter",
    "SyntheticEEGGenerator",
]
