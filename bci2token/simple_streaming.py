"""
Simple streaming interface for real-time BCI processing.

Provides basic streaming capabilities without complex dependencies.
"""

import time
import threading
import queue
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np

from .utils import validate_signal_shape, BCIError


class StreamingError(BCIError):
    """Error in streaming operations."""
    pass


@dataclass
class StreamConfig:
    """Configuration for streaming."""
    sampling_rate: int = 256
    buffer_size: int = 1024
    channels: int = 8
    processing_interval: float = 0.1  # seconds
    
    
class SimpleStreamer:
    """
    Simple real-time data streamer for BCI applications.
    
    Handles buffering, processing, and basic real-time operations
    without complex dependencies.
    """
    
    def __init__(self, config: StreamConfig, 
                 process_callback: Optional[Callable] = None):
        self.config = config
        self.process_callback = process_callback
        
        # Streaming state
        self.is_streaming = False
        self.data_buffer = queue.Queue(maxsize=config.buffer_size)
        self.processed_data = queue.Queue(maxsize=100)
        
        # Threading
        self.stream_thread = None
        self.process_thread = None
        self.stop_event = threading.Event()
        
        # Metrics
        self.samples_received = 0
        self.samples_processed = 0
        self.start_time = None
        
    def start_streaming(self):
        """Start the streaming process."""
        if self.is_streaming:
            raise StreamingError("Already streaming")
            
        self.is_streaming = True
        self.stop_event.clear()
        self.start_time = time.time()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop_streaming(self):
        """Stop the streaming process."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
            
        # Clear buffers
        while not self.data_buffer.empty():
            try:
                self.data_buffer.get_nowait()
            except queue.Empty:
                break
                
    def add_data(self, data: np.ndarray):
        """Add new data to the streaming buffer."""
        if not self.is_streaming:
            raise StreamingError("Not streaming")
            
        validate_signal_shape(data, self.config.channels)
        
        try:
            self.data_buffer.put_nowait(data)
            self.samples_received += data.shape[1]
        except queue.Full:
            raise StreamingError("Buffer overflow - data coming too fast")
            
    def get_processed_data(self, timeout: float = 0.1) -> Optional[Any]:
        """Get processed data from the output queue."""
        try:
            return self.processed_data.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _process_loop(self):
        """Main processing loop running in separate thread."""
        accumulated_data = []
        
        while not self.stop_event.is_set():
            try:
                # Get data with timeout
                data = self.data_buffer.get(timeout=self.config.processing_interval)
                accumulated_data.append(data)
                
                # Process when we have enough data
                if len(accumulated_data) >= 5:  # Process every 5 chunks
                    combined_data = np.concatenate(accumulated_data, axis=1)
                    
                    if self.process_callback:
                        try:
                            result = self.process_callback(combined_data)
                            self.processed_data.put_nowait(result)
                            self.samples_processed += combined_data.shape[1]
                        except Exception as e:
                            print(f"Processing error: {e}")
                    
                    accumulated_data = []
                    
            except queue.Empty:
                # No data available, continue
                continue
            except Exception as e:
                print(f"Stream processing error: {e}")
                break
                
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics."""
        if not self.start_time:
            return {}
            
        elapsed = time.time() - self.start_time
        
        return {
            'is_streaming': self.is_streaming,
            'elapsed_time': elapsed,
            'samples_received': self.samples_received,
            'samples_processed': self.samples_processed,
            'receive_rate': self.samples_received / elapsed if elapsed > 0 else 0,
            'process_rate': self.samples_processed / elapsed if elapsed > 0 else 0,
            'buffer_size': self.data_buffer.qsize(),
            'processed_queue_size': self.processed_data.qsize(),
        }


class MockDataGenerator:
    """Generate mock EEG data for testing streaming."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.time_index = 0
        
    def generate_chunk(self, duration: float = 0.1) -> np.ndarray:
        """Generate a chunk of mock EEG data."""
        n_samples = int(duration * self.config.sampling_rate)
        
        # Generate realistic-looking EEG data
        t = np.linspace(self.time_index, 
                       self.time_index + duration, 
                       n_samples)
        
        data = np.zeros((self.config.channels, n_samples))
        
        for ch in range(self.config.channels):
            # Alpha rhythm (8-12 Hz) with noise
            alpha = 2 * np.sin(2 * np.pi * (8 + ch * 0.5) * t)
            
            # Add some beta activity
            beta = 0.5 * np.sin(2 * np.pi * (20 + ch) * t)
            
            # Add noise
            noise = 0.3 * np.random.randn(n_samples)
            
            data[ch] = alpha + beta + noise
            
        self.time_index += duration
        return data


def create_simple_processor(config: StreamConfig):
    """Create a simple processing function for streaming."""
    
    def process_function(data: np.ndarray) -> Dict[str, Any]:
        """Simple processing: calculate band powers."""
        try:
            from scipy import signal
            
            # Calculate power spectral density
            freqs, psd = signal.welch(data, fs=config.sampling_rate, axis=1)
            
            # Calculate band powers
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            
            alpha_power = np.mean(psd[:, alpha_mask], axis=1)
            beta_power = np.mean(psd[:, beta_mask], axis=1)
            
            return {
                'timestamp': time.time(),
                'alpha_power': alpha_power,
                'beta_power': beta_power,
                'alpha_beta_ratio': alpha_power / (beta_power + 1e-8),
                'total_power': np.sum(psd, axis=1)
            }
            
        except ImportError:
            # Fallback without scipy
            return {
                'timestamp': time.time(),
                'rms_power': np.sqrt(np.mean(data**2, axis=1)),
                'mean_amplitude': np.mean(np.abs(data), axis=1)
            }
    
    return process_function