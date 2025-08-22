"""
Neural encoder models for brain signal to token decoding.

Implements various architectures including Transformer-based encoders,
CTC decoders, and diffusion models for high-accuracy brain decoding.
"""

try:
    import enhanced_mock_torch
    torch = enhanced_mock_torch
    nn = enhanced_mock_torch.nn
    F = enhanced_mock_torch.functional
except ImportError:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        import mock_torch
        torch = mock_torch.torch
        nn = mock_torch.torch.nn
        F = mock_torch.F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import warnings

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Transformers library not available. Some tokenizer features will be limited.")


@dataclass
class ModelConfig:
    """Configuration for brain-to-token models."""
    
    # Input configuration
    n_channels: int = 64
    sequence_length: int = 512  # Input sequence length (time points)
    sampling_rate: int = 256
    
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Output configuration
    vocab_size: int = 50257  # GPT-2 vocab size
    max_sequence_length: int = 128
    
    # Training configuration
    label_smoothing: float = 0.1
    ctc_blank_idx: int = 0


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class BrainSignalEncoder(nn.Module):
    """
    Encodes brain signals into high-dimensional representations.
    
    Uses convolutional layers followed by transformer architecture
    to capture both local temporal patterns and global dependencies.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            # Temporal convolution across channels
            nn.Conv1d(config.n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        # Project to model dimension
        self.projection = nn.Linear(256, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through brain signal encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            src_key_padding_mask: Mask for padded sequences
            
        Returns:
            Encoded representations of shape (batch, time, d_model)
        """
        # Apply convolutional feature extraction
        x = self.conv_layers(x)  # (batch, 256, time)
        
        # Transpose for transformer: (batch, time, features)
        x = x.transpose(1, 2)
        
        # Project to model dimension
        x = self.projection(x)  # (batch, time, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class CTCDecoder(nn.Module):
    """
    CTC (Connectionist Temporal Classification) decoder for sequence prediction.
    
    Fast inference with good accuracy for brain-to-text decoding.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.d_model, config.vocab_size + 1)  # +1 for blank
        self.ctc_loss = nn.CTCLoss(blank=config.ctc_blank_idx, reduction='mean')
        
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CTC decoder.
        
        Args:
            encoder_output: Encoded representations (batch, time, d_model)
            
        Returns:
            Token logits (batch, time, vocab_size + 1)
        """
        # Project to vocabulary space
        logits = self.output_projection(encoder_output)
        
        # Apply log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
        
    def decode_greedy(self, log_probs: torch.Tensor) -> List[List[int]]:
        """
        Greedy CTC decoding.
        
        Args:
            log_probs: Log probabilities (batch, time, vocab_size + 1)
            
        Returns:
            List of decoded token sequences
        """
        batch_size, seq_len, vocab_size = log_probs.shape
        
        # Get most likely tokens at each timestep
        predicted_tokens = torch.argmax(log_probs, dim=-1)  # (batch, time)
        
        decoded_sequences = []
        
        for batch_idx in range(batch_size):
            sequence = predicted_tokens[batch_idx].tolist()
            
            # Remove consecutive duplicates and blanks
            decoded = []
            prev_token = None
            
            for token in sequence:
                if token != self.config.ctc_blank_idx and token != prev_token:
                    decoded.append(token)
                prev_token = token
                
            decoded_sequences.append(decoded)
            
        return decoded_sequences


class DiffusionDecoder(nn.Module):
    """
    Diffusion-based decoder for high-accuracy token prediction.
    
    Uses diffusion process to iteratively refine token predictions,
    achieving higher accuracy at the cost of increased inference time.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Noise prediction network
        self.noise_predictor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.n_layers // 2
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Diffusion schedule
        self.register_buffer('betas', self._create_noise_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _create_noise_schedule(self, timesteps: int = 1000) -> torch.Tensor:
        """Create noise schedule for diffusion process."""
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
        
    def forward(self, encoder_output: torch.Tensor, 
                target_tokens: Optional[torch.Tensor] = None,
                timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through diffusion decoder.
        
        Args:
            encoder_output: Brain signal encoding (batch, time, d_model)
            target_tokens: Target token sequence for training
            timesteps: Diffusion timesteps
            
        Returns:
            Predicted noise or token logits
        """
        batch_size = encoder_output.size(0)
        
        if self.training and target_tokens is not None:
            # Training: predict noise added to target tokens
            if timesteps is None:
                timesteps = torch.randint(0, len(self.betas), (batch_size,), 
                                        device=encoder_output.device)
            
            # Add noise to target tokens
            noise = torch.randn_like(target_tokens, dtype=torch.float)
            sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
            
            # Reshape for broadcasting
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1)
            
            noisy_tokens = (sqrt_alphas_cumprod * target_tokens.float() + 
                           sqrt_one_minus_alphas_cumprod * noise)
            
            # Embed noisy tokens
            token_emb = self.token_embedding(target_tokens)
            
            # Predict noise
            noise_pred = self.noise_predictor(
                token_emb,
                encoder_output
            )
            
            return self.output_projection(noise_pred)
        else:
            # Inference: iterative denoising
            return self._sample(encoder_output)
            
    def _sample(self, encoder_output: torch.Tensor, 
                num_inference_steps: int = 50) -> torch.Tensor:
        """
        Sample tokens using diffusion process.
        
        Args:
            encoder_output: Brain signal encoding
            num_inference_steps: Number of denoising steps
            
        Returns:
            Sampled token logits
        """
        batch_size = encoder_output.size(0)
        seq_len = self.config.max_sequence_length
        
        # Start with pure noise
        tokens = torch.randn(batch_size, seq_len, self.config.d_model, 
                           device=encoder_output.device)
        
        # Denoising steps
        timesteps = torch.linspace(len(self.betas) - 1, 0, num_inference_steps, 
                                 dtype=torch.long, device=encoder_output.device)
        
        for t in timesteps:
            # Predict noise
            with torch.no_grad():
                noise_pred = self.noise_predictor(tokens, encoder_output)
                
                # Remove predicted noise
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                
                tokens = (1 / torch.sqrt(alpha)) * (
                    tokens - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
                )
                
                # Add noise for next step (except last step)
                if t > 0:
                    noise = torch.randn_like(tokens)
                    tokens = tokens + torch.sqrt(self.betas[t]) * noise
        
        # Project to vocabulary space
        return self.output_projection(tokens)


class BrainToTokenModel(nn.Module):
    """
    Complete brain-to-token model combining encoder and decoder.
    """
    
    def __init__(self, config: ModelConfig, decoder_type: str = 'ctc'):
        super().__init__()
        self.config = config
        self.decoder_type = decoder_type
        
        # Brain signal encoder
        self.encoder = BrainSignalEncoder(config)
        
        # Decoder
        if decoder_type == 'ctc':
            self.decoder = CTCDecoder(config)
        elif decoder_type == 'diffusion':
            self.decoder = DiffusionDecoder(config)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
            
    def forward(self, brain_signals: torch.Tensor,
                target_tokens: Optional[torch.Tensor] = None,
                signal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through complete model.
        
        Args:
            brain_signals: Brain signal tensor (batch, channels, time)
            target_tokens: Target tokens for training
            signal_mask: Mask for padded brain signals
            
        Returns:
            Token predictions or logits
        """
        # Encode brain signals
        encoded = self.encoder(brain_signals, src_key_padding_mask=signal_mask)
        
        # Decode to tokens
        output = self.decoder(encoded, target_tokens)
        
        return output
        
    def decode(self, brain_signals: torch.Tensor,
               signal_mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Decode brain signals to token sequences.
        
        Args:
            brain_signals: Brain signal tensor (batch, channels, time)
            signal_mask: Mask for padded brain signals
            
        Returns:
            List of decoded token sequences
        """
        self.eval()
        with torch.no_grad():
            if self.decoder_type == 'ctc':
                log_probs = self.forward(brain_signals, signal_mask=signal_mask)
                return self.decoder.decode_greedy(log_probs)
            else:
                # For diffusion decoder, return argmax of logits
                logits = self.forward(brain_signals, signal_mask=signal_mask)
                predicted_tokens = torch.argmax(logits, dim=-1)
                return predicted_tokens.tolist()