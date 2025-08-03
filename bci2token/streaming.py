"""
Real-time streaming decoder for live BCI applications.

Provides low-latency brain signal processing and token generation
for interactive brain-computer interfaces.
"""

import numpy as np
import threading
import time
import queue
from typing import Optional, Generator, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
from collections import deque
import warnings

from .decoder import BrainDecoder
from .llm_interface import LLMInterface


@dataclass
class StreamingConfig:
    """Configuration for real-time streaming."""
    
    buffer_duration: float = 2.0  # Buffer duration in seconds
    update_interval: float = 0.1  # Update interval in seconds
    confidence_threshold: float = 0.7  # Minimum confidence for output
    max_latency: float = 0.1  # Maximum allowed latency in seconds
    smoothing_window: int = 5  # Number of predictions to smooth over
    auto_punctuation: bool = True  # Automatically add punctuation
    word_boundary_detection: bool = True  # Detect word boundaries


class CircularBuffer:
    """Thread-safe circular buffer for signal data."""
    
    def __init__(self, max_size: int, n_channels: int):
        self.max_size = max_size
        self.n_channels = n_channels
        self.buffer = np.zeros((n_channels, max_size))
        self.head = 0
        self.size = 0
        self.lock = threading.Lock()
        
    def add_data(self, data: np.ndarray):
        """Add new data to buffer."""
        with self.lock:
            n_samples = data.shape[1]
            
            if n_samples >= self.max_size:
                # If data is larger than buffer, just take the end
                self.buffer = data[:, -self.max_size:]
                self.head = 0
                self.size = self.max_size
            else:
                # Add data to circular buffer
                end_idx = (self.head + n_samples) % self.max_size
                
                if end_idx > self.head:
                    # No wraparound
                    self.buffer[:, self.head:end_idx] = data
                else:
                    # Wraparound
                    split = self.max_size - self.head
                    self.buffer[:, self.head:] = data[:, :split]
                    self.buffer[:, :end_idx] = data[:, split:]
                    
                self.head = end_idx
                self.size = min(self.size + n_samples, self.max_size)
                
    def get_latest(self, n_samples: int) -> Optional[np.ndarray]:
        """Get latest n_samples from buffer."""
        with self.lock:
            if self.size < n_samples:
                return None
                
            start_idx = (self.head - n_samples) % self.max_size
            
            if start_idx < self.head:
                # No wraparound
                return self.buffer[:, start_idx:self.head].copy()
            else:
                # Wraparound
                part1 = self.buffer[:, start_idx:].copy()
                part2 = self.buffer[:, :self.head].copy()
                return np.concatenate([part1, part2], axis=1)


class PredictionSmoother:
    """Smooth predictions over time to reduce noise."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        
    def add_prediction(self, tokens: List[int], confidence: float):
        """Add new prediction to smoother."""
        self.predictions.append(tokens)
        self.confidences.append(confidence)
        
    def get_smoothed_prediction(self) -> Tuple[List[int], float]:
        """Get smoothed prediction."""
        if not self.predictions:
            return [], 0.0
            
        if len(self.predictions) == 1:
            return self.predictions[0], self.confidences[0]
            
        # Weight recent predictions more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.predictions)))
        weights = weights / np.sum(weights)
        
        # For tokens, use majority voting with confidence weighting
        all_tokens = []
        all_weights = []
        
        for tokens, conf, weight in zip(self.predictions, self.confidences, weights):
            all_tokens.extend(tokens)
            all_weights.extend([conf * weight] * len(tokens))
            
        # Get most confident tokens
        if not all_tokens:
            return [], 0.0
            
        # Simple approach: take most recent high-confidence prediction
        best_idx = np.argmax([conf * weight for conf, weight in zip(self.confidences, weights)])
        return list(self.predictions)[best_idx], list(self.confidences)[best_idx]


class StreamingDecoder:
    """
    Real-time streaming decoder for brain signals.
    
    Provides continuous processing of brain signals with low latency
    token generation and text output.
    """
    
    def __init__(self, 
                 decoder: BrainDecoder,
                 llm: LLMInterface,
                 config: Optional[StreamingConfig] = None):
        """
        Initialize streaming decoder.
        
        Args:
            decoder: Brain decoder instance
            llm: LLM interface for text generation
            config: Streaming configuration
        """
        self.decoder = decoder
        self.llm = llm
        self.config = config or StreamingConfig()
        
        # Calculate buffer parameters
        sampling_rate = decoder.sampling_rate
        buffer_size = int(self.config.buffer_duration * sampling_rate)
        window_size = int(2.0 * sampling_rate)  # 2 second processing windows
        
        # Initialize components
        self.signal_buffer = CircularBuffer(buffer_size, decoder.channels)
        self.prediction_smoother = PredictionSmoother(self.config.smoothing_window)
        
        # Processing parameters
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.update_samples = int(self.config.update_interval * sampling_rate)
        
        # State variables
        self.is_streaming = False
        self.last_processed_tokens = []
        self.accumulated_text = ""
        self.processing_thread = None
        
        # Callbacks
        self.token_callback: Optional[Callable[[List[int], float], None]] = None
        self.text_callback: Optional[Callable[[str], None]] = None
        
    def set_token_callback(self, callback: Callable[[List[int], float], None]):
        """Set callback for new tokens."""
        self.token_callback = callback
        
    def set_text_callback(self, callback: Callable[[str], None]):
        """Set callback for new text."""
        self.text_callback = callback
        
    def start_streaming(self):
        """Start streaming processing."""
        if self.is_streaming:
            warnings.warn("Streaming already active")
            return
            
        self.is_streaming = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_streaming(self):
        """Stop streaming processing."""
        self.is_streaming = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
    def add_data(self, signal_data: np.ndarray):
        """
        Add new signal data to processing buffer.
        
        Args:
            signal_data: New signal data (channels, timepoints)
        """
        if not self.is_streaming:
            warnings.warn("Streaming not active. Call start_streaming() first.")
            return
            
        self.signal_buffer.add_data(signal_data)
        
    def _processing_loop(self):
        """Main processing loop for streaming."""
        last_update = time.time()
        
        while self.is_streaming:
            current_time = time.time()
            
            # Check if it's time for an update
            if current_time - last_update >= self.config.update_interval:
                try:
                    self._process_latest_data()
                    last_update = current_time
                except Exception as e:
                    warnings.warn(f"Processing error: {e}")
                    
            # Small sleep to prevent busy waiting
            time.sleep(0.001)
            
    def _process_latest_data(self):
        """Process the latest data in the buffer."""
        # Get latest window of data
        latest_data = self.signal_buffer.get_latest(self.window_size)
        
        if latest_data is None:
            return  # Not enough data yet
            
        # Process with decoder
        start_time = time.time()
        
        try:
            result = self.decoder.decode_to_tokens(latest_data, return_confidence=True)
            tokens = result['tokens']
            confidence = np.mean(result['confidence']) if result['confidence'] else 0.0
            
        except Exception as e:
            warnings.warn(f"Decoding error: {e}")
            return
            
        processing_time = time.time() - start_time
        
        # Check latency constraint
        if processing_time > self.config.max_latency:
            warnings.warn(f"Processing latency ({processing_time:.3f}s) exceeds limit")
            
        # Apply smoothing
        self.prediction_smoother.add_prediction(tokens, confidence)
        smoothed_tokens, smoothed_confidence = self.prediction_smoother.get_smoothed_prediction()
        
        # Check confidence threshold
        if smoothed_confidence < self.config.confidence_threshold:
            return  # Confidence too low
            
        # Check for new tokens
        new_tokens = self._find_new_tokens(smoothed_tokens)
        
        if new_tokens:
            # Convert to text
            try:
                new_text = self.llm.tokens_to_text(new_tokens)
                new_text = self._post_process_text(new_text)
                
                # Update accumulated text
                self.accumulated_text += new_text
                
                # Trigger callbacks
                if self.token_callback:
                    self.token_callback(new_tokens, smoothed_confidence)
                    
                if self.text_callback:
                    self.text_callback(new_text)
                    
            except Exception as e:
                warnings.warn(f"Text conversion error: {e}")
                
        # Update last processed tokens
        self.last_processed_tokens = smoothed_tokens
        
    def _find_new_tokens(self, current_tokens: List[int]) -> List[int]:
        """Find new tokens compared to last processing."""
        if not self.last_processed_tokens:
            return current_tokens
            
        # Find longest common prefix
        common_length = 0
        min_length = min(len(current_tokens), len(self.last_processed_tokens))
        
        for i in range(min_length):
            if current_tokens[i] == self.last_processed_tokens[i]:
                common_length += 1
            else:
                break
                
        # Return new tokens
        return current_tokens[common_length:]
        
    def _post_process_text(self, text: str) -> str:
        """Apply post-processing to generated text."""
        if not text:
            return text
            
        # Auto-punctuation
        if self.config.auto_punctuation:
            text = self._add_punctuation(text)
            
        # Word boundary detection
        if self.config.word_boundary_detection:
            text = self._fix_word_boundaries(text)
            
        return text
        
    def _add_punctuation(self, text: str) -> str:
        """Add basic punctuation to text."""
        # Simple heuristics for punctuation
        text = text.strip()
        
        if not text:
            return text
            
        # Add periods at sentence boundaries (simple approach)
        if not text.endswith(('.', '!', '?', ':')):
            # Check if this looks like end of sentence
            words = text.split()
            if len(words) > 3:  # Reasonable sentence length
                text += '.'
                
        return text
        
    def _fix_word_boundaries(self, text: str) -> str:
        """Fix word boundary issues from tokenization."""
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Fix common tokenization artifacts
        text = text.replace(' ##', '')  # BERT-style subwords
        text = text.replace('##', '')
        
        return text
        
    def get_accumulated_text(self) -> str:
        """Get all accumulated text since streaming started."""
        return self.accumulated_text.strip()
        
    def clear_accumulated_text(self):
        """Clear accumulated text buffer."""
        self.accumulated_text = ""
        
    def get_status(self) -> Dict[str, Any]:
        """Get current streaming status."""
        return {
            'is_streaming': self.is_streaming,
            'buffer_size': self.signal_buffer.size,
            'max_buffer_size': self.signal_buffer.max_size,
            'accumulated_text_length': len(self.accumulated_text),
            'last_tokens_count': len(self.last_processed_tokens),
            'smoothing_window_size': len(self.prediction_smoother.predictions)
        }


class StreamingSession:
    """
    Context manager for streaming sessions.
    
    Provides convenient interface for managing streaming sessions
    with automatic cleanup.
    """
    
    def __init__(self, streaming_decoder: StreamingDecoder):
        self.decoder = streaming_decoder
        self.token_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
    def __enter__(self):
        # Set up callbacks to capture tokens and text
        self.decoder.set_token_callback(self._token_callback)
        self.decoder.set_text_callback(self._text_callback)
        
        # Start streaming
        self.decoder.start_streaming()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop streaming
        self.decoder.stop_streaming()
        
        # Clear callbacks
        self.decoder.set_token_callback(None)
        self.decoder.set_text_callback(None)
        
    def _token_callback(self, tokens: List[int], confidence: float):
        """Internal token callback."""
        self.token_queue.put((tokens, confidence))
        
    def _text_callback(self, text: str):
        """Internal text callback."""
        self.text_queue.put(text)
        
    def stream_tokens(self, timeout: float = 1.0) -> Generator[Tuple[List[int], float], None, None]:
        """
        Stream tokens as they become available.
        
        Args:
            timeout: Timeout for waiting for new tokens
            
        Yields:
            Tuples of (tokens, confidence)
        """
        while True:
            try:
                tokens, confidence = self.token_queue.get(timeout=timeout)
                yield tokens, confidence
            except queue.Empty:
                if not self.decoder.is_streaming:
                    break
                    
    def stream_text(self, timeout: float = 1.0) -> Generator[str, None, None]:
        """
        Stream text as it becomes available.
        
        Args:
            timeout: Timeout for waiting for new text
            
        Yields:
            New text strings
        """
        while True:
            try:
                text = self.text_queue.get(timeout=timeout)
                yield text
            except queue.Empty:
                if not self.decoder.is_streaming:
                    break
                    
    def add_signal_data(self, data: np.ndarray):
        """Add signal data to the streaming decoder."""
        self.decoder.add_data(data)
        
    def get_full_text(self) -> str:
        """Get all accumulated text."""
        return self.decoder.get_accumulated_text()