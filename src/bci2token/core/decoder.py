"""
Brain Signal Decoder - Core component for converting neural signals to tokens
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum

from ..models.architectures import ConformerCTC, DiffusionDecoder
from ..preprocessing.signal_processor import SignalProcessor
from ..privacy.differential_privacy import PrivacyEngine

logger = logging.getLogger(__name__)


class DecoderType(Enum):
    """Available decoder architectures"""
    CTC_CONFORMER = "ctc-conformer"
    DIFFUSION_INVERSE = "diffusion-inverse"
    TRANSFORMER_CTC = "transformer-ctc"


@dataclass
class DecoderConfig:
    """Configuration for brain decoder"""
    signal_type: str = "eeg"  # eeg, ecog, fnirs
    channels: int = 64
    sampling_rate: int = 256
    decoder_type: DecoderType = DecoderType.CTC_CONFORMER
    vocab_size: int = 50257  # GPT tokenizer size
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    privacy_epsilon: Optional[float] = None
    device: str = "auto"


class BrainDecoder(nn.Module):
    """
    Main brain signal decoder that converts EEG/ECoG signals to token logits
    compatible with language models.
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.device = self._setup_device()
        
        # Initialize signal processor
        self.signal_processor = SignalProcessor(
            signal_type=config.signal_type,
            channels=config.channels,
            sampling_rate=config.sampling_rate
        )
        
        # Initialize privacy engine if specified
        self.privacy_engine = None
        if config.privacy_epsilon is not None:
            self.privacy_engine = PrivacyEngine(
                epsilon=config.privacy_epsilon,
                delta=1e-5
            )
        
        # Initialize decoder model
        self.decoder_model = self._build_decoder()
        
        # Move to device
        self.to(self.device)
        
        logger.info(
            f"Initialized BrainDecoder with {config.decoder_type.value} "
            f"for {config.signal_type} signals ({config.channels} channels)"
        )
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder model based on configuration"""
        if self.config.decoder_type == DecoderType.CTC_CONFORMER:
            return ConformerCTC(
                input_dim=self.config.channels,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                vocab_size=self.config.vocab_size,
                dropout=self.config.dropout
            )
        elif self.config.decoder_type == DecoderType.DIFFUSION_INVERSE:
            return DiffusionDecoder(
                input_dim=self.config.channels,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                vocab_size=self.config.vocab_size,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"Unsupported decoder type: {self.config.decoder_type}")
    
    def preprocess_signals(
        self, 
        raw_signals: np.ndarray,
        apply_privacy: bool = True
    ) -> torch.Tensor:
        """
        Preprocess raw brain signals
        
        Args:
            raw_signals: Raw neural signals (channels, time_points)
            apply_privacy: Whether to apply differential privacy
            
        Returns:
            Preprocessed signals ready for decoding
        """
        # Signal preprocessing pipeline
        processed = self.signal_processor.process(raw_signals)
        
        # Apply differential privacy if configured
        if apply_privacy and self.privacy_engine is not None:
            processed = self.privacy_engine.add_noise(processed)
        
        # Convert to tensor
        tensor = torch.from_numpy(processed).float()
        
        return tensor.to(self.device)
    
    def forward(
        self, 
        signals: torch.Tensor, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through decoder
        
        Args:
            signals: Preprocessed brain signals (batch, channels, time)
            return_attention: Whether to return attention weights
            
        Returns:
            Token logits (and optionally attention weights)
        """
        if self.config.decoder_type in [DecoderType.CTC_CONFORMER, DecoderType.TRANSFORMER_CTC]:
            return self.decoder_model(signals, return_attention=return_attention)
        else:
            # Diffusion-based decoder
            return self.decoder_model.sample(signals)
    
    def decode_to_tokens(
        self, 
        raw_signals: np.ndarray,
        confidence_threshold: float = 0.7,
        beam_width: int = 1
    ) -> List[int]:
        """
        Decode brain signals to token IDs
        
        Args:
            raw_signals: Raw neural signals
            confidence_threshold: Minimum confidence for token prediction
            beam_width: Beam search width for decoding
            
        Returns:
            List of predicted token IDs
        """
        self.eval()
        with torch.no_grad():
            # Preprocess signals
            signals = self.preprocess_signals(raw_signals)
            
            # Add batch dimension if needed
            if signals.dim() == 2:
                signals = signals.unsqueeze(0)
            
            # Forward pass
            logits = self.forward(signals)
            
            # Decode tokens based on model type
            if self.config.decoder_type in [DecoderType.CTC_CONFORMER, DecoderType.TRANSFORMER_CTC]:
                tokens = self._ctc_decode(logits, confidence_threshold, beam_width)
            else:
                tokens = self._diffusion_decode(logits, confidence_threshold)
            
            return tokens
    
    def _ctc_decode(
        self, 
        logits: torch.Tensor, 
        confidence_threshold: float,
        beam_width: int
    ) -> List[int]:
        """CTC decoding with beam search"""
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        if beam_width == 1:
            # Greedy decoding
            predicted_tokens = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]
            
            # Filter by confidence
            valid_predictions = confidences > confidence_threshold
            tokens = predicted_tokens[valid_predictions].cpu().numpy().tolist()
        else:
            # Beam search decoding (simplified implementation)
            tokens = self._beam_search_decode(probs, beam_width, confidence_threshold)
        
        # Remove CTC blank tokens (assuming 0 is blank)
        tokens = [t for t in tokens if t != 0]
        
        return tokens
    
    def _diffusion_decode(
        self, 
        samples: torch.Tensor, 
        confidence_threshold: float
    ) -> List[int]:
        """Decode from diffusion samples"""
        # Convert samples to token probabilities
        probs = torch.softmax(samples, dim=-1)
        
        # Get most likely tokens
        predicted_tokens = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]
        
        # Filter by confidence
        valid_predictions = confidences > confidence_threshold
        tokens = predicted_tokens[valid_predictions].cpu().numpy().tolist()
        
        return tokens
    
    def _beam_search_decode(
        self, 
        probs: torch.Tensor, 
        beam_width: int,
        confidence_threshold: float
    ) -> List[int]:
        """Simplified beam search implementation"""
        # For demonstration - simplified beam search
        # In practice, would use more sophisticated CTC beam search
        batch_size, seq_len, vocab_size = probs.shape
        
        # Greedy approximation for now
        predicted_tokens = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]
        
        # Filter by confidence
        valid_predictions = confidences[0] > confidence_threshold
        tokens = predicted_tokens[0][valid_predictions].cpu().numpy().tolist()
        
        return tokens
    
    def get_attention_weights(self, signals: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights for interpretability"""
        if hasattr(self.decoder_model, 'get_attention_weights'):
            return self.decoder_model.get_attention_weights(signals)
        return None
    
    def calibrate(
        self, 
        calibration_data: List[Tuple[np.ndarray, List[int]]],
        num_epochs: int = 10
    ) -> Dict[str, float]:
        """
        Calibrate decoder on user-specific data
        
        Args:
            calibration_data: List of (signals, target_tokens) pairs
            num_epochs: Number of calibration epochs
            
        Returns:
            Calibration metrics
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for signals, target_tokens in calibration_data:
                # Preprocess signals
                processed_signals = self.preprocess_signals(signals, apply_privacy=False)
                
                # Convert targets to tensor
                targets = torch.tensor(target_tokens, device=self.device)
                
                # Forward pass
                logits = self.forward(processed_signals)
                
                # Compute loss (CTC or standard cross-entropy)
                if self.config.decoder_type in [DecoderType.CTC_CONFORMER, DecoderType.TRANSFORMER_CTC]:
                    loss = self._ctc_loss(logits, targets)
                else:
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
            logger.info(f"Calibration epoch {epoch + 1}/{num_epochs}, loss: {epoch_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "calibration_loss": avg_loss,
            "num_epochs": num_epochs,
            "num_samples": len(calibration_data)
        }
    
    def _ctc_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss"""
        # Simplified CTC loss - in practice would use torch.nn.CTCLoss
        log_probs = torch.log_softmax(logits, dim=-1)
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
        target_lengths = torch.full((targets.size(0),), targets.size(1), dtype=torch.long)
        
        ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
        return ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
    
    def save_model(self, path: str) -> None:
        """Save model state and configuration"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'signal_processor_state': self.signal_processor.get_state()
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'BrainDecoder':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        decoder = cls(checkpoint['config'])
        decoder.load_state_dict(checkpoint['model_state_dict'])
        decoder.signal_processor.load_state(checkpoint['signal_processor_state'])
        logger.info(f"Model loaded from {path}")
        return decoder
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "decoder_type": self.config.decoder_type.value,
            "signal_type": self.config.signal_type,
            "channels": self.config.channels,
            "sampling_rate": self.config.sampling_rate,
            "vocab_size": self.config.vocab_size,
            "total_parameters": num_params,
            "trainable_parameters": num_trainable,
            "device": str(self.device),
            "privacy_enabled": self.privacy_engine is not None,
            "privacy_epsilon": self.config.privacy_epsilon
        }