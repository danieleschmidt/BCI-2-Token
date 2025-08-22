"""
Main BrainDecoder class for converting brain signals to tokens.

Integrates preprocessing, neural models, and privacy protection
into a unified interface for brain-to-token decoding.
"""

try:
    # First try to import the enhanced mock
    import enhanced_mock_torch
    torch = enhanced_mock_torch
    F = enhanced_mock_torch.functional
except ImportError:
    try:
        # Try real PyTorch
        import torch
        import torch.nn.functional as F
    except ImportError:
        # Fall back to basic mock
        import mock_torch
        torch = mock_torch.torch
        F = mock_torch.F
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings

from .preprocessing import SignalPreprocessor, PreprocessingConfig
from .models import BrainToTokenModel, ModelConfig
from .privacy import PrivacyEngine
from .utils import validate_signal_shape, SignalProcessingError
from .monitoring import get_monitor


class _DummyContext:
    """Dummy context manager for when performance timer not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def _dummy_context():
    return _DummyContext()


class BrainDecoder:
    """
    Main interface for brain signal to token decoding.
    
    Handles the complete pipeline from raw brain signals to LLM-compatible tokens,
    including preprocessing, neural decoding, and privacy protection.
    """
    
    def __init__(self,
                 signal_type: str = 'eeg',
                 channels: int = 64,
                 sampling_rate: int = 256,
                 model_type: str = 'ctc',
                 privacy_epsilon: Optional[float] = None,
                 model_path: Optional[Union[str, Path]] = None):
        """
        Initialize brain decoder.
        
        Args:
            signal_type: Type of brain signals ('eeg', 'ecog', 'fnirs')
            channels: Number of recording channels
            sampling_rate: Sampling rate in Hz
            model_type: Neural model type ('ctc', 'diffusion')
            privacy_epsilon: Differential privacy budget (None to disable)
            model_path: Path to pretrained model weights
        """
        self.signal_type = signal_type
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.model_type = model_type
        self.privacy_epsilon = privacy_epsilon
        
        # Initialize preprocessing
        self.preproc_config = PreprocessingConfig(
            sampling_rate=sampling_rate,
            # Adjust preprocessing based on signal type
            lowpass_freq=40.0 if signal_type == 'eeg' else 100.0,
            highpass_freq=0.5 if signal_type == 'eeg' else 1.0,
            apply_ica=signal_type == 'eeg',  # ICA mainly useful for EEG
            apply_car=True
        )
        self.preprocessor = SignalPreprocessor(self.preproc_config)
        
        # Initialize model
        self.model_config = ModelConfig(
            n_channels=channels,
            sampling_rate=sampling_rate,
            sequence_length=int(2.0 * sampling_rate),  # 2 second windows
        )
        self.model = BrainToTokenModel(self.model_config, decoder_type=model_type)
        
        # Load pretrained weights if provided
        if model_path is not None:
            self.load_model(model_path)
            
        # Initialize privacy protection
        self.privacy_engine = None
        if privacy_epsilon is not None:
            from .privacy import PrivacyEngine
            self.privacy_engine = PrivacyEngine(
                epsilon=privacy_epsilon,
                delta=1e-5,
                signal_channels=channels,
                sampling_rate=sampling_rate
            )
            
        # Set to evaluation mode by default
        self.model.eval()
        
    def preprocess_signals(self, 
                          raw_signals: np.ndarray,
                          channel_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Preprocess raw brain signals with comprehensive error handling.
        
        Args:
            raw_signals: Raw signal data (channels, timepoints)
            channel_names: Optional channel names
            
        Returns:
            Dictionary with preprocessed data and metadata
            
        Raises:
            SignalProcessingError: If signal validation fails
        """
        monitor = get_monitor()
        
        try:
            # Validate input signal
            validate_signal_shape(raw_signals, self.channels, min_timepoints=10)
            
            # Log preprocessing start
            monitor.logger.debug(
                'Decoder',
                f'Preprocessing signal: {raw_signals.shape}',
                {'channels': raw_signals.shape[0], 'timepoints': raw_signals.shape[1]}
            )
            
            # Apply preprocessing pipeline
            with monitor.performance_timer('preprocessing') if hasattr(monitor, 'performance_timer') else _dummy_context():
                processed = self.preprocessor.preprocess(raw_signals, channel_names)
            
            # Apply privacy protection if enabled
            if self.privacy_engine is not None:
                try:
                    processed['processed_data'] = self.privacy_engine.add_noise(
                        processed['processed_data']
                    )
                    processed['epochs'] = self.privacy_engine.add_noise(
                        processed['epochs']
                    )
                    
                    monitor.logger.debug(
                        'Privacy',
                        f'Applied DP noise: ε={self.privacy_epsilon}',
                        {'epsilon': self.privacy_epsilon}
                    )
                except Exception as e:
                    monitor.log_error('Privacy', e, {'signal_shape': raw_signals.shape})
                    raise SignalProcessingError(f"Privacy protection failed: {e}") from e
            
            # Log successful preprocessing
            monitor.logger.debug(
                'Decoder',
                f'Preprocessing completed: {len(processed["epochs"])} epochs',
                {'input_shape': raw_signals.shape, 'epochs_created': len(processed["epochs"])}
            )
            
            return processed
            
        except Exception as e:
            monitor.log_error('Decoder', e, {'operation': 'preprocess_signals'})
            if isinstance(e, SignalProcessingError):
                raise
            else:
                raise SignalProcessingError(f"Preprocessing failed: {e}") from e
        
    def decode_to_tokens(self, 
                        brain_signals: np.ndarray,
                        return_confidence: bool = False) -> Union[List[int], Dict[str, Any]]:
        """
        Decode brain signals directly to token IDs.
        
        Args:
            brain_signals: Brain signal data (channels, timepoints)
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of token IDs or dict with tokens and confidence
        """
        # Preprocess signals
        processed = self.preprocess_signals(brain_signals)
        epochs = processed['epochs']  # (n_epochs, channels, timepoints)
        
        if len(epochs) == 0:
            warnings.warn("No valid epochs found in signal data")
            return [] if not return_confidence else {'tokens': [], 'confidence': []}
            
        # Convert to torch tensor
        signal_tensor = torch.FloatTensor(epochs)
        
        # Decode using neural model
        with torch.no_grad():
            if self.model_type == 'ctc':
                log_probs = self.model(signal_tensor)
                token_sequences = self.model.decode(signal_tensor)
                
                # For multiple epochs, concatenate sequences
                all_tokens = []
                confidences = []
                
                for i, tokens in enumerate(token_sequences):
                    all_tokens.extend(tokens)
                    # Calculate confidence from log probabilities
                    epoch_log_probs = log_probs[i]
                    max_probs = torch.max(torch.exp(epoch_log_probs), dim=-1)[0]
                    confidences.extend(max_probs.tolist())
                    
            else:  # diffusion
                logits = self.model(signal_tensor)
                # Take average across epochs
                mean_logits = torch.mean(logits, dim=0)
                all_tokens = torch.argmax(mean_logits, dim=-1).tolist()
                
                # Calculate confidence from softmax probabilities
                probs = torch.softmax(mean_logits, dim=-1)
                confidences = torch.max(probs, dim=-1)[0].tolist()
                
        if return_confidence:
            return {
                'tokens': all_tokens,
                'confidence': confidences,
                'num_epochs': len(epochs)
            }
        else:
            return all_tokens
            
    def decode_to_logits(self, brain_signals: np.ndarray) -> np.ndarray:
        """
        Decode brain signals to token logits for LLM integration.
        
        Args:
            brain_signals: Brain signal data (channels, timepoints)
            
        Returns:
            Token logits array (sequence_length, vocab_size)
        """
        # Preprocess signals
        processed = self.preprocess_signals(brain_signals)
        epochs = processed['epochs']
        
        if len(epochs) == 0:
            # Return zero logits if no valid epochs
            return np.zeros((self.model_config.max_sequence_length, 
                           self.model_config.vocab_size))
            
        # Convert to torch tensor
        signal_tensor = torch.FloatTensor(epochs)
        
        # Get logits from model
        with torch.no_grad():
            if self.model_type == 'ctc':
                log_probs = self.model(signal_tensor)
                # Convert log probabilities to logits (remove log)
                # Average across epochs and remove CTC blank token
                mean_log_probs = torch.mean(log_probs, dim=0)  # (time, vocab_size + 1)
                logits = mean_log_probs[:, 1:]  # Remove blank token
                
            else:  # diffusion
                logits = self.model(signal_tensor)
                # Average across epochs
                logits = torch.mean(logits, dim=0)  # (sequence_length, vocab_size)
                
        return logits.numpy()
        
    def calibrate(self, 
                  calibration_signals: List[np.ndarray],
                  calibration_texts: List[str],
                  tokenizer_name: str = 'gpt2') -> Dict[str, float]:
        """
        Calibrate decoder on user-specific data.
        
        Args:
            calibration_signals: List of brain signal recordings
            calibration_texts: Corresponding text labels
            tokenizer_name: Tokenizer to use for text encoding
            
        Returns:
            Calibration metrics
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            raise ImportError("transformers library required for calibration")
            
        if len(calibration_signals) != len(calibration_texts):
            raise ValueError("Number of signals and texts must match")
            
        # Set model to training mode
        self.model.train()
        
        # Simple fine-tuning loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        total_loss = 0.0
        n_batches = 0
        
        for signals, text in zip(calibration_signals, calibration_texts):
            # Preprocess signals
            processed = self.preprocess_signals(signals)
            epochs = processed['epochs']
            
            if len(epochs) == 0:
                continue
                
            # Tokenize text
            tokens = tokenizer.encode(text, return_tensors='pt')
            if tokens.size(1) > self.model_config.max_sequence_length:
                tokens = tokens[:, :self.model_config.max_sequence_length]
                
            # Forward pass
            signal_tensor = torch.FloatTensor(epochs)
            
            if self.model_type == 'ctc':
                log_probs = self.model(signal_tensor)
                # CTC loss expects (time, batch, vocab)
                log_probs = log_probs.transpose(0, 1)
                
                # Create target lengths
                input_lengths = torch.full((len(epochs),), log_probs.size(0), dtype=torch.long)
                target_lengths = torch.full((len(epochs),), tokens.size(1), dtype=torch.long)
                
                # Repeat targets for each epoch
                targets = tokens.repeat(len(epochs), 1).flatten()
                
                loss = self.model.decoder.ctc_loss(
                    log_probs, targets, input_lengths, target_lengths
                )
            else:
                # For diffusion, use simple cross-entropy
                logits = self.model(signal_tensor, target_tokens=tokens.repeat(len(epochs), 1))
                targets = tokens.repeat(len(epochs), 1)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        # Return to evaluation mode
        self.model.eval()
        
        avg_loss = total_loss / max(n_batches, 1)
        
        return {
            'calibration_loss': avg_loss,
            'num_samples': len(calibration_signals),
            'num_batches': n_batches
        }
        
    def save_model(self, path: Union[str, Path]):
        """Save model weights and configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config.__dict__,
            'preproc_config': self.preproc_config.__dict__,
            'signal_type': self.signal_type,
            'channels': self.channels,
            'sampling_rate': self.sampling_rate,
            'model_type': self.model_type,
            'privacy_epsilon': self.privacy_epsilon
        }
        
        torch.save(state_dict, path)
        
    def load_model(self, path: Union[str, Path]):
        """Load model weights and configuration."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update configurations if saved
        if 'model_config' in checkpoint:
            for key, value in checkpoint['model_config'].items():
                setattr(self.model_config, key, value)
                
        if 'preproc_config' in checkpoint:
            for key, value in checkpoint['preproc_config'].items():
                setattr(self.preproc_config, key, value)
                
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'signal_type': self.signal_type,
            'channels': self.channels,
            'sampling_rate': self.sampling_rate,
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.model_config.vocab_size,
            'sequence_length': self.model_config.sequence_length,
            'privacy_enabled': self.privacy_engine is not None,
            'privacy_epsilon': self.privacy_epsilon
        }