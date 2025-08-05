#!/usr/bin/env python3
"""
Real-time streaming demo for BCI-2-Token framework.

Demonstrates live brain signal processing with simulated device.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bci2token import BrainDecoder, LLMInterface
from bci2token.streaming import StreamingDecoder, StreamingConfig
from bci2token.devices import create_device, DeviceConfig


def main():
    print("BCI-2-Token Real-Time Streaming Demo")
    print("=" * 40)
    
    # Set up simulated device
    print("1. Setting up simulated EEG device...")
    device_config = DeviceConfig(
        device_type='simulated',
        sampling_rate=256,
        n_channels=8
    )
    
    device = create_device('simulated', device_config)
    
    # Connect to device
    print("2. Connecting to device...")
    if not device.connect():
        print("Failed to connect to device")
        return 1
        
    print("   ✓ Device connected")
    
    # Initialize brain decoder
    print("3. Initializing brain decoder...")
    decoder = BrainDecoder(
        signal_type='eeg',
        channels=8,
        sampling_rate=256,
        model_type='ctc'
    )
    
    # Initialize LLM interface
    llm = LLMInterface(model_name='gpt2')
    
    # Set up streaming configuration
    streaming_config = StreamingConfig(
        buffer_duration=2.0,
        update_interval=0.2,
        confidence_threshold=0.5,  # Lower threshold for demo
        smoothing_window=3
    )
    
    # Create streaming decoder
    streaming_decoder = StreamingDecoder(decoder, llm, streaming_config)
    
    print("4. Starting real-time processing...")
    print("   Think your message... (streaming for 10 seconds)")
    print("   Decoded text will appear below:")
    print("-" * 40)
    
    # Track collected data and results
    collected_chunks = []
    decoded_text = ""
    
    try:
        # Set up data collection
        def on_new_data(data):
            collected_chunks.append(data.copy())
            streaming_decoder.add_data(data)
            
        device.set_data_callback(on_new_data)
        
        # Start device streaming
        device.start_streaming()
        
        # Use streaming session
        with streaming_decoder.start_session() as session:
            start_time = time.time()
            duration = 10.0  # Stream for 10 seconds
            
            # Stream text as it becomes available
            try:
                for text_chunk in session.stream_text(timeout=1.0):
                    print(text_chunk, end='', flush=True)
                    decoded_text += text_chunk
                    
                    # Check if time is up
                    if time.time() - start_time > duration:
                        break
                        
            except KeyboardInterrupt:
                print("\n\nStreaming interrupted by user")
                
    finally:
        # Clean up
        device.stop_streaming()
        device.disconnect()
        
    print("\n" + "-" * 40)
    print("5. Streaming session completed")
    
    # Show results
    total_signal_time = sum(chunk.shape[1] for chunk in collected_chunks) / 256
    print(f"   Processed {len(collected_chunks)} data chunks")
    print(f"   Total signal duration: {total_signal_time:.1f} seconds")
    print(f"   Final decoded text: '{decoded_text}'")
    
    # Show streaming status
    status = streaming_decoder.get_status()
    print(f"   Buffer utilization: {status['buffer_size']}/{status['max_buffer_size']}")
    
    print("\n✓ Streaming demo completed successfully!")
    

if __name__ == '__main__':
    main()