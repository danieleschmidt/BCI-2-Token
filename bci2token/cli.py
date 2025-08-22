"""
Command-line interface for BCI-2-Token framework.

Provides easy-to-use CLI commands for brain signal decoding, training,
and real-time streaming applications.
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any

try:
    from . import BrainDecoder, LLMInterface, StreamingDecoder
    _HAS_CORE = True
except ImportError:
    BrainDecoder = None
    LLMInterface = None
    StreamingDecoder = None
    _HAS_CORE = False

try:
    from .devices import create_device, DeviceConfig
    _HAS_DEVICES = True
except ImportError:
    create_device = None
    DeviceConfig = None
    _HAS_DEVICES = False

try:
    from .streaming import StreamingConfig
    _HAS_STREAMING = True
except ImportError:
    StreamingConfig = None
    _HAS_STREAMING = False

try:
    from .training import BrainDecoderTrainer, TrainingConfig, BrainTextDataset
    _HAS_TRAINING = True
except ImportError:
    BrainDecoderTrainer = None
    TrainingConfig = None
    BrainTextDataset = None
    _HAS_TRAINING = False

try:
    from .models import ModelConfig
    _HAS_MODELS = True
except ImportError:
    ModelConfig = None
    _HAS_MODELS = False

try:
    from .preprocessing import PreprocessingConfig
    _HAS_PREPROCESSING = True
except ImportError:
    PreprocessingConfig = None
    _HAS_PREPROCESSING = False


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='bci2token',
        description='Brain-Computer Interface to Token Translator'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode brain signals to text')
    decode_parser.add_argument('input_file', help='Input brain signal file (.npy)')
    decode_parser.add_argument('--output', '-o', help='Output text file')
    decode_parser.add_argument('--signal-type', default='eeg', choices=['eeg', 'ecog', 'fnirs'])
    decode_parser.add_argument('--channels', type=int, default=64)
    decode_parser.add_argument('--sampling-rate', type=int, default=256)
    decode_parser.add_argument('--model-type', default='ctc', choices=['ctc', 'diffusion'])
    decode_parser.add_argument('--model-path', help='Path to pretrained model')
    decode_parser.add_argument('--llm-model', default='gpt2', help='LLM model name')
    decode_parser.add_argument('--privacy-epsilon', type=float, help='Differential privacy budget')
    decode_parser.add_argument('--confidence-threshold', type=float, default=0.7)
    
    # Stream command
    stream_parser = subparsers.add_parser('stream', help='Real-time streaming decoding')
    stream_parser.add_argument('--device-type', default='simulated', 
                              choices=['openbci', 'emotiv', 'lsl', 'simulated'])
    stream_parser.add_argument('--device-port', default='/dev/ttyUSB0')
    stream_parser.add_argument('--signal-type', default='eeg', choices=['eeg', 'ecog', 'fnirs'])
    stream_parser.add_argument('--channels', type=int, default=8)
    stream_parser.add_argument('--sampling-rate', type=int, default=256)
    stream_parser.add_argument('--model-type', default='ctc', choices=['ctc', 'diffusion'])
    stream_parser.add_argument('--model-path', help='Path to pretrained model')
    stream_parser.add_argument('--llm-model', default='gpt2')
    stream_parser.add_argument('--duration', type=float, default=30.0, help='Streaming duration in seconds')
    stream_parser.add_argument('--output', help='Output text file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train brain decoder model')
    train_parser.add_argument('data_dir', help='Directory containing training data')
    train_parser.add_argument('--output-dir', default='./models', help='Output directory for trained models')
    train_parser.add_argument('--model-type', default='ctc', choices=['ctc', 'diffusion'])
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--learning-rate', type=float, default=1e-4)
    train_parser.add_argument('--privacy-epsilon', type=float, help='DP privacy budget')
    train_parser.add_argument('--config', help='JSON config file')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument('--model-path', help='Show information about specific model')
    info_parser.add_argument('--health', action='store_true', help='Run health diagnostics')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    
    return parser


def cmd_decode(args) -> int:
    """Handle decode command."""
    try:
        # Load brain signal data
        print(f"Loading brain signals from {args.input_file}...")
        brain_data = np.load(args.input_file)
        
        if brain_data.ndim != 2:
            print(f"Error: Expected 2D data (channels, timepoints), got {brain_data.ndim}D")
            return 1
            
        print(f"Loaded signal: {brain_data.shape[0]} channels, {brain_data.shape[1]} timepoints")
        
        # Initialize decoder
        decoder = BrainDecoder(
            signal_type=args.signal_type,
            channels=args.channels,
            sampling_rate=args.sampling_rate,
            model_type=args.model_type,
            privacy_epsilon=args.privacy_epsilon,
            model_path=args.model_path
        )
        
        # Initialize LLM interface
        llm = LLMInterface(model_name=args.llm_model)
        
        print("Decoding brain signals...")
        
        # Decode to tokens with confidence
        result = decoder.decode_to_tokens(brain_data, return_confidence=True)
        tokens = result['tokens']
        confidence = result.get('confidence', [])
        
        if not tokens:
            print("No tokens decoded from brain signals")
            return 0
            
        # Filter by confidence threshold
        if confidence:
            avg_confidence = np.mean(confidence)
            print(f"Average confidence: {avg_confidence:.3f}")
            
            if avg_confidence < args.confidence_threshold:
                print(f"Warning: Confidence ({avg_confidence:.3f}) below threshold ({args.confidence_threshold})")
        
        # Convert to text
        text = llm.tokens_to_text(tokens)
        
        print(f"\nDecoded text: '{text}'")
        print(f"Tokens: {tokens}")
        
        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(text)
            print(f"Saved to {args.output}")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_stream(args) -> int:
    """Handle stream command."""
    try:
        # Set up device
        device_config = DeviceConfig(
            device_type=args.device_type,
            port=args.device_port,
            sampling_rate=args.sampling_rate,
            n_channels=args.channels
        )
        
        device = create_device(args.device_type, device_config)
        
        print(f"Connecting to {args.device_type} device...")
        if not device.connect():
            print("Failed to connect to device")
            return 1
            
        print("Device connected successfully")
        
        # Initialize decoder
        decoder = BrainDecoder(
            signal_type=args.signal_type,
            channels=args.channels,
            sampling_rate=args.sampling_rate,
            model_type=args.model_type,
            model_path=args.model_path
        )
        
        # Initialize LLM interface
        llm = LLMInterface(model_name=args.llm_model)
        
        # Set up streaming
        streaming_config = StreamingConfig(
            buffer_duration=2.0,
            update_interval=0.1,
            confidence_threshold=0.7
        )
        
        streaming_decoder = StreamingDecoder(decoder, llm, streaming_config)
        
        print(f"Starting real-time decoding for {args.duration} seconds...")
        print("Think your message...")
        
        # Collect decoded text
        decoded_text = ""
        
        try:
            # Start device streaming
            device.start_streaming()
            
            # Set up data callback
            def on_new_data(data):
                streaming_decoder.add_data(data)
                
            device.set_data_callback(on_new_data)
            
            # Start streaming decoder
            with streaming_decoder.start_session() as session:
                start_time = time.time()
                
                # Stream for specified duration
                for text_chunk in session.stream_text(timeout=1.0):
                    print(text_chunk, end='', flush=True)
                    decoded_text += text_chunk
                    
                    # Check duration
                    if time.time() - start_time > args.duration:
                        break
                        
        finally:
            device.stop_streaming()
            device.disconnect()
            
        print(f"\n\nFinal decoded text: '{decoded_text}'")
        
        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(decoded_text)
            print(f"Saved to {args.output}")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_train(args) -> int:
    """Handle train command."""
    try:
        print(f"Training brain decoder on data from {args.data_dir}")
        
        # Load training configuration
        if args.config:
            with open(args.config) as f:
                config_dict = json.load(f)
            training_config = TrainingConfig(**config_dict)
        else:
            training_config = TrainingConfig(
                model_type=args.model_type,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_dp=args.privacy_epsilon is not None,
                dp_epsilon=args.privacy_epsilon or 1.0
            )
            
        # Create model config
        model_config = ModelConfig()
        
        # Initialize trainer
        trainer = BrainDecoderTrainer(
            model_config=model_config,
            training_config=training_config,
            output_dir=args.output_dir
        )
        
        # TODO: Load actual training data
        print("Error: Training data loading not implemented yet")
        print("Please implement data loading from the specified directory")
        
        return 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_info(args) -> int:
    """Handle info command."""
    try:
        print("BCI-2-Token System Information")
        print("=" * 40)
        
        # System info
        try:
            import enhanced_mock_torch
            torch = enhanced_mock_torch
            print(f"PyTorch: Using enhanced mock implementation")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA devices: {torch.cuda.device_count()}")
        except ImportError:
            try:
                import torch
                print(f"PyTorch version: {torch.__version__}")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"CUDA devices: {torch.cuda.device_count()}")
            except ImportError:
                try:
                    import mock_torch
                    torch = mock_torch.torch
                    print("PyTorch: Using basic mock implementation")
                    print("CUDA available: False")
                except ImportError:
                    print("PyTorch: Not available (no mock implementation)")
            
        # Check health if requested
        if hasattr(args, 'health') and args.health:
            from .health import run_comprehensive_diagnostics
            print("\nHealth Diagnostics:")
            print("-" * 20)
            health_results = run_comprehensive_diagnostics()
            for check_name, result in health_results.items():
                status = "✓" if result.level.value == "healthy" else "✗"
                print(f"  {status} {check_name}: {result.message}")
            
        # Try to import optional dependencies
        deps = {
            'mne': 'Signal processing',
            'transformers': 'LLM tokenizers',
            'tiktoken': 'OpenAI tokenizers',
            'opacus': 'Differential privacy',
            'pylsl': 'Lab Streaming Layer',
            'serial': 'Serial device support'
        }
        
        print("\nDependency Status:")
        for dep, desc in deps.items():
            try:
                __import__(dep)
                status = "✓ Available"
            except ImportError:
                status = "✗ Not installed"
            print(f"  {dep:12} ({desc:20}): {status}")
            
        # Model info if specified
        if args.model_path:
            print(f"\nModel Information: {args.model_path}")
            decoder = BrainDecoder(model_path=args.model_path)
            info = decoder.get_model_info()
            
            for key, value in info.items():
                print(f"  {key}: {value}")
                
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_test(args) -> int:
    """Handle test command."""
    try:
        print("Running BCI-2-Token system tests...")
        
        # Check if core components are available  
        if BrainDecoder is None:
            print("⚠️  Core components not available (missing PyTorch)")
            print("   Running limited functionality tests...")
            
            # Test preprocessing
            print("1. Testing signal preprocessing...")
            try:
                from .preprocessing import PreprocessingConfig, SignalPreprocessor
                config = PreprocessingConfig(sampling_rate=256)
                preprocessor = SignalPreprocessor(config)
                print("   ✓ Preprocessor initialized successfully")
            except ImportError as e:
                print(f"   ✗ Preprocessor not available: {e}")
                return 1
                
            # Test health monitoring
            print("2. Testing health monitoring...")
            from .health import run_comprehensive_diagnostics
            health_results = run_comprehensive_diagnostics()
            print(f"   ✓ Health checks completed ({len(health_results)} checks)")
            
            print("\nLimited tests passed! Install PyTorch for full functionality.")
            return 0
        
        # Basic functionality test
        print("1. Testing basic decoder initialization...")
        decoder = BrainDecoder(signal_type='eeg', channels=8, sampling_rate=256)
        print("   ✓ Decoder initialized successfully")
        
        # Test with synthetic data
        print("2. Testing with synthetic brain signals...")
        test_signal = np.random.randn(8, 512)  # 8 channels, 2 seconds at 256 Hz
        
        try:
            tokens = decoder.decode_to_tokens(test_signal)
            print(f"   ✓ Decoded {len(tokens)} tokens")
        except Exception as e:
            print(f"   ✗ Decoding failed: {e}")
            return 1
            
        # Test LLM interface
        print("3. Testing LLM interface...")
        llm = LLMInterface(model_name='gpt2')
        
        if tokens:
            try:
                text = llm.tokens_to_text(tokens)
                print(f"   ✓ Converted to text: '{text[:50]}...'")
            except Exception as e:
                print(f"   ✗ Text conversion failed: {e}")
                return 1
        else:
            print("   - No tokens to convert")
            
        # Test device simulation
        print("4. Testing simulated device...")
        device_config = DeviceConfig(device_type='simulated', n_channels=8)
        device = create_device('simulated', device_config)
        
        if device.connect():
            print("   ✓ Simulated device connected")
            device.disconnect()
        else:
            print("   ✗ Failed to connect to simulated device")
            return 1
            
        print("\nAll tests passed! ✓")
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Import time here to avoid import at module level
    import time
    
    # Route to appropriate command handler
    if args.command == 'decode':
        return cmd_decode(args)
    elif args.command == 'stream':
        return cmd_stream(args)
    elif args.command == 'train':
        return cmd_train(args)
    elif args.command == 'info':
        return cmd_info(args)
    elif args.command == 'test':
        return cmd_test(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())