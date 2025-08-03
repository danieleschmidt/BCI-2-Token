"""
LLM integration interface for converting brain-decoded tokens to text.

Supports multiple tokenizers and language models including GPT, LLaMA, Claude, etc.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
import warnings

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers library not available. LLM integration will be limited.")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class LLMInterface:
    """
    Interface for integrating brain-decoded tokens with language models.
    
    Supports various tokenizers and provides utilities for converting between
    brain-predicted logits and natural language text.
    """
    
    def __init__(self, 
                 model_name: str = 'gpt2',
                 tokenizer_name: Optional[str] = None,
                 device: str = 'cpu',
                 max_length: int = 128):
        """
        Initialize LLM interface.
        
        Args:
            model_name: Name of the language model ('gpt2', 'gpt-4', 'llama', etc.)
            tokenizer_name: Tokenizer name (defaults to model_name)
            device: Device for model inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = device
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Initialize model (optional for some use cases)
        self.model = None
        if self._should_load_model():
            self.model = self._load_model()
            
    def _load_tokenizer(self):
        """Load appropriate tokenizer based on model name."""
        if 'gpt-4' in self.model_name.lower() or 'gpt-3.5' in self.model_name.lower():
            # Use tiktoken for OpenAI models
            if HAS_TIKTOKEN:
                if 'gpt-4' in self.model_name.lower():
                    return tiktoken.encoding_for_model("gpt-4")
                else:
                    return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                warnings.warn("tiktoken not available, falling back to GPT-2 tokenizer")
                if HAS_TRANSFORMERS:
                    return AutoTokenizer.from_pretrained('gpt2')
                else:
                    raise ImportError("Neither tiktoken nor transformers available")
                    
        elif 'claude' in self.model_name.lower():
            # Claude uses a similar tokenizer to GPT-4
            if HAS_TIKTOKEN:
                return tiktoken.encoding_for_model("gpt-4")
            else:
                warnings.warn("tiktoken not available for Claude, using GPT-2 tokenizer")
                if HAS_TRANSFORMERS:
                    return AutoTokenizer.from_pretrained('gpt2')
                else:
                    raise ImportError("No tokenizer library available")
                    
        elif 'llama' in self.model_name.lower():
            if HAS_TRANSFORMERS:
                # Try to load LLaMA tokenizer
                try:
                    return AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
                except:
                    warnings.warn("LLaMA tokenizer not available, using GPT-2")
                    return AutoTokenizer.from_pretrained('gpt2')
            else:
                raise ImportError("transformers library required for LLaMA")
                
        else:
            # Default to transformers AutoTokenizer
            if HAS_TRANSFORMERS:
                try:
                    return AutoTokenizer.from_pretrained(self.tokenizer_name)
                except:
                    warnings.warn(f"Could not load {self.tokenizer_name}, using GPT-2")
                    return AutoTokenizer.from_pretrained('gpt2')
            else:
                raise ImportError("transformers library required for tokenizer")
                
    def _should_load_model(self) -> bool:
        """Determine if we should load the actual language model."""
        # Only load model for local inference
        local_models = ['gpt2', 'distilgpt2', 'microsoft/DialoGPT-medium']
        return any(model in self.model_name for model in local_models)
        
    def _load_model(self):
        """Load language model for local inference."""
        if not HAS_TRANSFORMERS:
            return None
            
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            warnings.warn(f"Could not load model {self.model_name}: {e}")
            return None
            
    def tokens_to_text(self, tokens: List[int]) -> str:
        """
        Convert token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        if hasattr(self.tokenizer, 'decode'):
            # Transformers tokenizer
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        elif hasattr(self.tokenizer, 'decode_single_token_bytes'):
            # tiktoken tokenizer
            text_bytes = b''.join(self.tokenizer.decode_single_token_bytes(token) 
                                 for token in tokens)
            return text_bytes.decode('utf-8', errors='replace')
        else:
            raise ValueError("Unknown tokenizer type")
            
    def text_to_tokens(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        if hasattr(self.tokenizer, 'encode'):
            # Both transformers and tiktoken have encode method
            if HAS_TRANSFORMERS and hasattr(self.tokenizer, 'return_tensors'):
                # Transformers tokenizer
                return self.tokenizer.encode(text, max_length=self.max_length, 
                                           truncation=True)
            else:
                # tiktoken tokenizer
                tokens = self.tokenizer.encode(text)
                return tokens[:self.max_length]  # Manual truncation
        else:
            raise ValueError("Unknown tokenizer type")
            
    def logits_to_text(self, 
                       logits: np.ndarray,
                       temperature: float = 1.0,
                       top_k: int = 50,
                       top_p: float = 0.9) -> str:
        """
        Convert logits to text using sampling strategies.
        
        Args:
            logits: Token logits array (sequence_length, vocab_size)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated text string
        """
        if logits.ndim != 2:
            raise ValueError("Logits must be 2D (sequence_length, vocab_size)")
            
        # Convert to torch tensor for easier manipulation
        logits_tensor = torch.FloatTensor(logits)
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits_tensor = logits_tensor / temperature
            
        # Sample tokens
        sampled_tokens = []
        
        for step_logits in logits_tensor:
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(step_logits, top_k)
                step_logits = torch.full_like(step_logits, float('-inf'))
                step_logits.scatter_(-1, top_k_indices, top_k_logits)
                
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                step_logits[indices_to_remove] = float('-inf')
                
            # Sample from the filtered distribution
            probs = torch.softmax(step_logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            sampled_tokens.append(token_id)
            
        return self.tokens_to_text(sampled_tokens)
        
    def complete_text(self, 
                      brain_logits: np.ndarray,
                      prompt: str = "",
                      max_new_tokens: int = 50) -> str:
        """
        Use brain logits as initial context for text completion.
        
        Args:
            brain_logits: Brain-decoded logits (sequence_length, vocab_size)
            prompt: Optional text prompt to prepend
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Completed text
        """
        if self.model is None:
            # Fallback: just decode brain logits directly
            return self.logits_to_text(brain_logits)
            
        # Convert brain logits to initial tokens
        brain_tokens = torch.argmax(torch.FloatTensor(brain_logits), dim=-1).tolist()
        
        # Encode prompt if provided
        prompt_tokens = []
        if prompt:
            prompt_tokens = self.text_to_tokens(prompt)
            
        # Combine prompt and brain tokens
        input_tokens = prompt_tokens + brain_tokens
        input_ids = torch.tensor([input_tokens]).to(self.device)
        
        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode generated tokens
        generated_tokens = outputs[0][len(input_tokens):].tolist()
        return self.tokens_to_text(generated_tokens)
        
    def evaluate_perplexity(self, 
                           brain_logits: np.ndarray, 
                           target_text: str) -> float:
        """
        Evaluate how well brain logits match target text.
        
        Args:
            brain_logits: Brain-decoded logits
            target_text: Ground truth text
            
        Returns:
            Perplexity score (lower is better)
        """
        target_tokens = self.text_to_tokens(target_text)
        
        if len(target_tokens) > brain_logits.shape[0]:
            target_tokens = target_tokens[:brain_logits.shape[0]]
        elif len(target_tokens) < brain_logits.shape[0]:
            # Pad target tokens
            pad_token = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
            target_tokens.extend([pad_token] * (brain_logits.shape[0] - len(target_tokens)))
            
        # Calculate cross-entropy loss
        logits_tensor = torch.FloatTensor(brain_logits)
        target_tensor = torch.LongTensor(target_tokens)
        
        log_probs = torch.log_softmax(logits_tensor, dim=-1)
        nll_loss = torch.nn.functional.nll_loss(log_probs, target_tensor, reduction='mean')
        
        # Convert to perplexity
        perplexity = torch.exp(nll_loss).item()
        
        return perplexity
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size of the tokenizer."""
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'n_vocab'):
            return self.tokenizer.n_vocab
        else:
            # Estimate by trying to decode a range of token IDs
            max_token = 0
            for i in range(100000, 0, -1000):
                try:
                    self.tokens_to_text([i])
                    max_token = i
                    break
                except:
                    continue
            return max_token + 1000  # Add buffer
            
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about the loaded tokenizer."""
        return {
            'model_name': self.model_name,
            'tokenizer_name': self.tokenizer_name,
            'vocab_size': self.get_vocab_size(),
            'max_length': self.max_length,
            'has_model': self.model is not None,
            'tokenizer_type': type(self.tokenizer).__name__
        }