"""
BCI-2-Token Basic Usage Example

This example demonstrates the complete workflow of the BCI-2-Token system,
from signal acquisition to text generation with privacy protection.
"""

import numpy as np
import torch
from typing import List, Dict, Any

# BCI-2-Token imports
from bci2token import BrainDecoder, LLMInterface, PrivacyEngine
from bci2token.core.decoder import DecoderConfig, DecoderType
from bci2token.streaming.realtime import StreamingDecoder
from bci2token.preprocessing.signal_processor import SignalProcessor
from bci2token.agents.orchestrator import AgentOrchestrator
from bci2token.agents.base_agent import AgentContext
from monitoring.metrics import get_metrics_collector, monitor_performance


def generate_sample_eeg_data() -> np.ndarray:
    """
    Generate realistic sample EEG data for demonstration
    
    Returns:
        EEG signals array of shape (channels, time_points)
    """
    print("🧠 Generating sample EEG data...")
    
    # Parameters
    channels = 64
    duration_seconds = 4.0
    sampling_rate = 256
    time_points = int(duration_seconds * sampling_rate)
    
    # Time vector
    time = np.linspace(0, duration_seconds, time_points)
    
    # Initialize signals array
    signals = np.zeros((channels, time_points))
    
    # Simulate thinking "Hello world" - add realistic brain wave patterns
    for ch in range(channels):
        # Alpha waves (8-13 Hz) - relaxed awareness
        alpha = 0.5 * np.sin(2 * np.pi * 10 * time + np.random.uniform(0, 2*np.pi))
        
        # Beta waves (13-30 Hz) - active thinking
        beta = 0.3 * np.sin(2 * np.pi * 20 * time + np.random.uniform(0, 2*np.pi))
        
        # Gamma waves (30-100 Hz) - cognitive processing
        gamma = 0.2 * np.sin(2 * np.pi * 40 * time + np.random.uniform(0, 2*np.pi))
        
        # Add some theta (4-8 Hz) for memory processes
        theta = 0.25 * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
        
        # Noise (artifact simulation)
        noise = 0.1 * np.random.randn(time_points)
        
        # Combine all components
        signals[ch] = alpha + beta + gamma + theta + noise
        
        # Add spatial variation (different channels have different patterns)
        spatial_modifier = np.sin(ch * np.pi / channels)
        signals[ch] *= (0.8 + 0.4 * spatial_modifier)
    
    print(f"✅ Generated EEG data: {signals.shape} (channels × time points)")
    return signals


@monitor_performance("bci_demo")
def basic_decoder_example():
    """
    Demonstrate basic brain signal decoding to tokens
    """
    print("\n" + "="*60)
    print("🧠 BCI-2-Token Basic Decoder Example")
    print("="*60)
    
    # Initialize metrics collector
    metrics = get_metrics_collector()
    
    # 1. Create decoder configuration
    print("\n📋 Configuring BCI decoder...")
    config = DecoderConfig(
        signal_type="eeg",
        channels=64,
        sampling_rate=256,
        decoder_type=DecoderType.CTC_CONFORMER,
        vocab_size=50257,  # GPT tokenizer size
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        privacy_epsilon=1.0,  # Enable differential privacy
        device="cpu"  # Use CPU for demo
    )
    
    # 2. Initialize the brain decoder
    print("🤖 Initializing brain decoder...")
    decoder = BrainDecoder(config)
    print(f"✅ Decoder initialized: {decoder.get_model_info()['total_parameters']:,} parameters")
    
    # 3. Generate sample brain signals
    brain_signals = generate_sample_eeg_data()
    
    # 4. Decode brain signals to tokens
    print("\n🔄 Decoding brain signals to tokens...")
    
    with metrics.measure_time("inference_duration"):
        tokens = decoder.decode_to_tokens(
            brain_signals,
            confidence_threshold=0.5,
            beam_width=1
        )
    
    print(f"🎯 Decoded tokens: {tokens}")
    print(f"📊 Number of tokens: {len(tokens)}")
    
    # 5. Get attention weights for interpretability
    if len(tokens) > 0:
        print("\n🔍 Analyzing attention patterns...")
        processed_signals = decoder.preprocess_signals(brain_signals)
        attention_weights = decoder.get_attention_weights(processed_signals.unsqueeze(0))
        
        if attention_weights is not None:
            print(f"✅ Attention weights shape: {attention_weights.shape}")
        else:
            print("ℹ️  Attention weights not available for this model type")
    
    # 6. Record performance metrics
    metrics.record_inference_metrics({
        "inference_latency_ms": 150.0,  # Example metrics
        "preprocessing_time_ms": 25.0,
        "model_forward_time_ms": 100.0,
        "postprocessing_time_ms": 25.0,
        "total_memory_mb": 512.0,
        "cpu_usage_percent": 45.0,
        "throughput_tokens_per_second": len(tokens) / 0.15 if len(tokens) > 0 else 0,
        "accuracy_score": 0.87,
        "confidence_score": 0.75
    })
    
    return tokens, decoder


def privacy_protection_example():
    """
    Demonstrate differential privacy protection of neural signals
    """
    print("\n" + "="*60)
    print("🔒 Privacy Protection Example")
    print("="*60)
    
    # Generate sample data
    brain_signals = generate_sample_eeg_data()
    
    # Create decoder with privacy protection
    print("\n🛡️  Setting up privacy-preserving decoder...")
    privacy_config = DecoderConfig(
        signal_type="eeg",
        channels=64,
        sampling_rate=256,
        privacy_epsilon=1.0,  # Privacy budget
        device="cpu"
    )
    
    private_decoder = BrainDecoder(privacy_config)
    
    # Process signals with and without privacy
    print("🔄 Processing signals with privacy protection...")
    
    # Without privacy (for comparison)
    signals_no_privacy = private_decoder.preprocess_signals(
        brain_signals, 
        apply_privacy=False
    )
    
    # With privacy protection
    signals_with_privacy = private_decoder.preprocess_signals(
        brain_signals, 
        apply_privacy=True
    )
    
    # Calculate privacy impact
    signal_difference = torch.mean(torch.abs(signals_no_privacy - signals_with_privacy))
    snr_loss = 20 * torch.log10(
        torch.std(signals_no_privacy) / torch.std(signals_with_privacy - signals_no_privacy)
    )
    
    print(f"🔍 Privacy impact analysis:")
    print(f"   • Signal difference (MAE): {signal_difference:.6f}")
    print(f"   • SNR loss: {snr_loss:.2f} dB")
    print(f"   • Privacy epsilon: {privacy_config.privacy_epsilon}")
    
    # Decode both versions
    tokens_no_privacy = private_decoder.decode_to_tokens(brain_signals, apply_privacy=False)
    tokens_with_privacy = private_decoder.decode_to_tokens(brain_signals, apply_privacy=True)
    
    print(f"\n📊 Decoding results:")
    print(f"   • Without privacy: {len(tokens_no_privacy)} tokens")
    print(f"   • With privacy: {len(tokens_with_privacy)} tokens")
    
    return private_decoder


def llm_integration_example():
    """
    Demonstrate integration with language models
    """
    print("\n" + "="*60)
    print("🤝 LLM Integration Example")
    print("="*60)
    
    # Simulate token output from decoder
    sample_tokens = [15496, 995, 318, 257, 1332, 2646]  # "Hello world is a test message"
    
    print(f"🎯 Sample decoded tokens: {sample_tokens}")
    
    # Create mock LLM interface
    print("\n🤖 Initializing LLM interface...")
    
    class MockLLMInterface:
        """Mock LLM interface for demonstration"""
        
        def __init__(self, model_name: str):
            self.model_name = model_name
            # Simple token-to-text mapping for demo
            self.token_map = {
                15496: "Hello",
                995: " world",
                318: " is",
                257: " a", 
                1332: " test",
                2646: " message"
            }
        
        def tokens_to_text(self, tokens: List[int]) -> str:
            """Convert tokens to text"""
            text_parts = []
            for token in tokens:
                if token in self.token_map:
                    text_parts.append(self.token_map[token])
                else:
                    text_parts.append(f"<unk_{token}>")
            return "".join(text_parts)
        
        def complete_text(self, partial_text: str) -> str:
            """Complete partial text"""
            if "test message" in partial_text:
                return partial_text + " from the brain-computer interface!"
            return partial_text
    
    llm = MockLLMInterface("mock-gpt-4")
    
    # Convert tokens to text
    print("🔄 Converting tokens to text...")
    decoded_text = llm.tokens_to_text(sample_tokens)
    print(f"📝 Decoded text: '{decoded_text}'")
    
    # Complete the text
    print("\n🧠 Completing text with LLM...")
    completed_text = llm.complete_text(decoded_text)
    print(f"✨ Completed text: '{completed_text}'")
    
    return completed_text


def streaming_demo():
    """
    Demonstrate real-time streaming capabilities
    """
    print("\n" + "="*60)
    print("🌊 Real-time Streaming Demo")
    print("="*60)
    
    print("🚀 Simulating real-time brain signal streaming...")
    
    # Create streaming configuration
    config = DecoderConfig(
        signal_type="eeg",
        channels=64,
        sampling_rate=256,
        device="cpu"
    )
    
    decoder = BrainDecoder(config)
    
    # Simulate streaming data processing
    chunk_size = 256  # 1 second of data at 256 Hz
    num_chunks = 5
    
    print(f"📊 Processing {num_chunks} chunks of {chunk_size} samples each...")
    
    all_tokens = []
    
    for chunk_idx in range(num_chunks):
        print(f"\n📦 Processing chunk {chunk_idx + 1}/{num_chunks}...")
        
        # Generate chunk of brain signals
        chunk_signals = np.random.randn(64, chunk_size)
        
        # Add some realistic patterns
        time = np.linspace(chunk_idx, chunk_idx + 1, chunk_size)
        for ch in range(64):
            alpha = 0.3 * np.sin(2 * np.pi * 10 * time)
            chunk_signals[ch] += alpha
        
        # Decode chunk
        chunk_tokens = decoder.decode_to_tokens(
            chunk_signals,
            confidence_threshold=0.6
        )
        
        all_tokens.extend(chunk_tokens)
        print(f"   🎯 Chunk tokens: {chunk_tokens}")
        print(f"   📈 Total tokens so far: {len(all_tokens)}")
        
        # Simulate processing delay
        import time
        time.sleep(0.1)  # 100ms processing time
    
    print(f"\n✅ Streaming complete! Total tokens decoded: {len(all_tokens)}")
    return all_tokens


def agent_orchestration_demo():
    """
    Demonstrate AI agent orchestration for development workflow
    """
    print("\n" + "="*60)
    print("🤖 AI Agent Orchestration Demo")
    print("="*60)
    
    # Create agent context
    context = AgentContext(
        project_root="/tmp/bci_demo",
        current_branch="feature/demo",
        requirements={"accuracy": ">90%", "latency": "<100ms"},
        architecture={"decoder_type": "conformer-ctc", "privacy": "enabled"}
    )
    
    print("🎭 Setting up AI agent orchestrator...")
    orchestrator = AgentOrchestrator(context)
    
    # Simulate workflow stages
    print("\n📋 Workflow stages:")
    for i, stage in enumerate(orchestrator.workflow_stages, 1):
        print(f"   {i}. {stage.name}: {', '.join(stage.agents)}")
    
    print(f"\n📊 Total registered agents: {len(orchestrator.agents)}")
    print("🎯 This would normally execute the full AI-powered development workflow")
    print("   including requirements analysis, architecture design, implementation,")
    print("   testing, security review, performance optimization, and deployment.")
    
    return orchestrator


def main():
    """
    Main demonstration function
    """
    print("🧠 BCI-2-Token Comprehensive Demo")
    print("==================================")
    print("This demo showcases the complete BCI-2-Token system capabilities:")
    print("• Brain signal decoding to tokens")
    print("• Privacy-preserving neural processing")
    print("• LLM integration for text generation")
    print("• Real-time streaming capabilities")
    print("• AI-powered development workflow")
    print("• Comprehensive monitoring and metrics")
    
    try:
        # 1. Basic decoding example
        tokens, decoder = basic_decoder_example()
        
        # 2. Privacy protection example
        private_decoder = privacy_protection_example()
        
        # 3. LLM integration example
        completed_text = llm_integration_example()
        
        # 4. Streaming demo
        streaming_tokens = streaming_demo()
        
        # 5. Agent orchestration demo
        orchestrator = agent_orchestration_demo()
        
        # 6. Generate final metrics report
        print("\n" + "="*60)
        print("📊 Final Metrics Report")
        print("="*60)
        
        metrics = get_metrics_collector()
        summary = metrics.get_metrics_summary()
        
        print(f"🎯 Total requests processed: {summary['system']['total_requests']}")
        print(f"⏱️  System uptime: {summary['system']['uptime_seconds']:.1f} seconds")
        print(f"📈 Requests per second: {summary['system']['requests_per_second']:.2f}")
        print(f"🧮 Metrics collected: {len(summary['recent_metrics'])} types")
        
        print("\n✅ Demo completed successfully!")
        print("🚀 BCI-2-Token system is ready for production deployment!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()