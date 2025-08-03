"""
Training scripts and utilities for brain-to-token models.

Provides training loops, optimization, and model management functionality
for brain signal decoding models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import warnings
from tqdm import tqdm

from .models import BrainToTokenModel, ModelConfig
from .privacy import PrivacyEngine
from .preprocessing import SignalPreprocessor, PreprocessingConfig

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers library not available. Limited tokenizer support.")


@dataclass 
class TrainingConfig:
    """Configuration for model training."""
    
    # Model parameters
    model_type: str = 'ctc'  # 'ctc' or 'diffusion'
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    
    # Scheduling
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'exponential'
    warmup_steps: int = 1000
    
    # Privacy
    use_dp: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    
    # Validation
    validation_split: float = 0.2
    validation_interval: int = 10  # epochs
    early_stopping_patience: int = 20
    
    # Checkpointing
    save_interval: int = 20
    max_checkpoints: int = 5
    
    # Curriculum learning
    use_curriculum: bool = False
    curriculum_stages: List[Tuple[str, int]] = None
    
    # Data augmentation
    use_augmentation: bool = True
    noise_std: float = 0.1
    time_stretch_factor: float = 0.1


class BrainTextDataset(Dataset):
    """Dataset for brain signals paired with text labels."""
    
    def __init__(self, 
                 brain_signals: List[np.ndarray],
                 texts: List[str],
                 tokenizer_name: str = 'gpt2',
                 max_text_length: int = 128,
                 preprocessor: Optional[SignalPreprocessor] = None):
        """
        Initialize dataset.
        
        Args:
            brain_signals: List of brain signal arrays (channels, timepoints)
            texts: Corresponding text labels
            tokenizer_name: Name of tokenizer to use
            max_text_length: Maximum text sequence length
            preprocessor: Optional signal preprocessor
        """
        assert len(brain_signals) == len(texts), "Number of signals and texts must match"
        
        self.brain_signals = brain_signals
        self.texts = texts
        self.max_text_length = max_text_length
        self.preprocessor = preprocessor
        
        # Initialize tokenizer
        if HAS_TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ImportError("transformers library required for dataset")
            
        # Preprocess signals if preprocessor provided
        if self.preprocessor:
            self.processed_signals = []
            for signal in brain_signals:
                processed = self.preprocessor.preprocess(signal)
                # Use first epoch if multiple epochs generated
                if len(processed['epochs']) > 0:
                    self.processed_signals.append(processed['epochs'][0])
                else:
                    # Fallback to processed data
                    self.processed_signals.append(processed['processed_data'])
        else:
            self.processed_signals = brain_signals
            
    def __len__(self) -> int:
        return len(self.brain_signals)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get brain signal
        signal = torch.FloatTensor(self.processed_signals[idx])
        
        # Tokenize text
        text = self.texts[idx]
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_text_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)
        
        return {
            'brain_signal': signal,
            'text_tokens': tokens,
            'text': text
        }


class BrainDecoderTrainer:
    """
    Trainer for brain-to-token decoder models.
    
    Handles training loop, validation, checkpointing, and optimization.
    """
    
    def __init__(self, 
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 output_dir: Union[str, Path] = './models'):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration  
            output_dir: Directory for saving models
        """
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = BrainToTokenModel(model_config, decoder_type=training_config.model_type)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize privacy engine if needed
        self.privacy_engine = None
        if training_config.use_dp:
            self.privacy_engine = PrivacyEngine(
                epsilon=training_config.dp_epsilon,
                delta=training_config.dp_delta,
                signal_channels=model_config.n_channels,
                sampling_rate=model_config.sampling_rate
            )
            
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.training_config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.num_epochs
            )
        elif self.training_config.scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return None
            
    def train(self, 
              train_dataset: BrainTextDataset,
              val_dataset: Optional[BrainTextDataset] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Training results dictionary
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
        # Training loop
        for epoch in range(self.training_config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = None
            if val_loader and epoch % self.training_config.validation_interval == 0:
                val_loss = self._validate_epoch(val_loader)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.training_config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Save checkpoint
            if epoch % self.training_config.save_interval == 0:
                self._save_checkpoint()
                
            # Log progress
            log_str = f"Epoch {epoch+1}/{self.training_config.num_epochs}, "
            log_str += f"Train Loss: {train_loss:.4f}"
            if val_loss:
                log_str += f", Val Loss: {val_loss:.4f}"
            print(log_str)
            
            # Store history
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
            
        return {
            'final_train_loss': train_loss,
            'best_val_loss': self.best_val_loss,
            'num_epochs_trained': self.current_epoch + 1,
            'training_history': self.training_history
        }
        
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}") as pbar:
            for batch in pbar:
                # Move to device
                brain_signals = batch['brain_signal'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                
                # Data augmentation
                if self.training_config.use_augmentation:
                    brain_signals = self._augment_signals(brain_signals)
                    
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.training_config.model_type == 'ctc':
                    log_probs = self.model(brain_signals)
                    loss = self._compute_ctc_loss(log_probs, text_tokens)
                else:  # diffusion
                    loss = self._compute_diffusion_loss(brain_signals, text_tokens)
                    
                # Backward pass
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.privacy_engine:
                    self._apply_dp_to_gradients()
                    
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.grad_clip_norm
                )
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / num_batches
        
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                brain_signals = batch['brain_signal'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                
                if self.training_config.model_type == 'ctc':
                    log_probs = self.model(brain_signals)
                    loss = self._compute_ctc_loss(log_probs, text_tokens)
                else:
                    loss = self._compute_diffusion_loss(brain_signals, text_tokens)
                    
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def _compute_ctc_loss(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss."""
        batch_size, seq_len, vocab_size = log_probs.shape
        
        # Remove padding tokens from targets
        target_lengths = []
        flattened_targets = []
        
        for i in range(batch_size):
            # Find non-padding tokens (assuming 0 is padding)
            non_pad_mask = targets[i] != 0
            non_pad_tokens = targets[i][non_pad_mask]
            
            target_lengths.append(len(non_pad_tokens))
            flattened_targets.extend(non_pad_tokens.tolist())
            
        # Create input lengths (all sequences use full length)
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        flattened_targets = torch.tensor(flattened_targets, dtype=torch.long)
        
        # Transpose for CTC: (time, batch, vocab)
        log_probs = log_probs.transpose(0, 1)
        
        # Compute CTC loss
        return self.model.decoder.ctc_loss(
            log_probs, flattened_targets, input_lengths, target_lengths
        )
        
    def _compute_diffusion_loss(self, brain_signals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss."""
        # For diffusion models, the loss is computed inside the model
        output = self.model(brain_signals, target_tokens=targets)
        
        # Simple cross-entropy loss for now
        # In practice, you'd want the proper diffusion objective
        loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        return loss(output.view(-1, output.size(-1)), targets.view(-1))
        
    def _augment_signals(self, signals: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to brain signals."""
        if not self.training_config.use_augmentation:
            return signals
            
        # Add Gaussian noise
        noise = torch.randn_like(signals) * self.training_config.noise_std
        signals = signals + noise
        
        # Time stretching (simple resampling)
        if np.random.random() < 0.3:  # 30% chance
            stretch_factor = 1.0 + np.random.uniform(
                -self.training_config.time_stretch_factor,
                self.training_config.time_stretch_factor
            )
            # Simple implementation - in practice use proper resampling
            if stretch_factor != 1.0:
                new_length = int(signals.size(-1) * stretch_factor)
                signals = torch.nn.functional.interpolate(
                    signals.unsqueeze(1),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                
                # Crop or pad to original length
                if new_length > signals.size(-1):
                    signals = signals[:, :, :signals.size(-1)]
                elif new_length < signals.size(-1):
                    pad_size = signals.size(-1) - new_length
                    signals = torch.nn.functional.pad(signals, (0, pad_size))
                    
        return signals
        
    def _apply_dp_to_gradients(self):
        """Apply differential privacy to gradients."""
        if not self.privacy_engine:
            return
            
        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad
                
        # Apply DP noise
        noisy_gradients = self.privacy_engine.add_noise_to_gradients(gradients)
        
        # Update model gradients
        for name, param in self.model.named_parameters():
            if name in noisy_gradients and noisy_gradients[name] is not None:
                param.grad = noisy_gradients[name]
                
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': asdict(self.model_config),
            'training_config': asdict(self.training_config),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        checkpoints = list(self.output_dir.glob('checkpoint_epoch_*.pt'))
        
        if len(checkpoints) > self.training_config.max_checkpoints:
            # Sort by epoch number and remove oldest
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for old_checkpoint in checkpoints[:-self.training_config.max_checkpoints]:
                old_checkpoint.unlink()
                
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
    def evaluate_model(self, test_dataset: BrainTextDataset) -> Dict[str, float]:
        """Evaluate model on test dataset."""
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.model.eval()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                brain_signals = batch['brain_signal'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                
                # Compute loss
                if self.training_config.model_type == 'ctc':
                    log_probs = self.model(brain_signals)
                    loss = self._compute_ctc_loss(log_probs, text_tokens)
                    
                    # Decode predictions for accuracy
                    predicted_sequences = self.model.decode(brain_signals)
                    
                    # Calculate token accuracy (simplified)
                    for i, pred_seq in enumerate(predicted_sequences):
                        target_seq = text_tokens[i][text_tokens[i] != 0].tolist()
                        correct_tokens += sum(1 for p, t in zip(pred_seq, target_seq) if p == t)
                        total_tokens += len(target_seq)
                        
                else:  # diffusion
                    loss = self._compute_diffusion_loss(brain_signals, text_tokens)
                    
                total_loss += loss.item()
                
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return {
            'test_loss': avg_loss,
            'token_accuracy': accuracy,
            'perplexity': np.exp(avg_loss)
        }