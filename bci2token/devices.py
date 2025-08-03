"""
Device drivers for common EEG/ECoG hardware.

Provides unified interface for various brain signal acquisition devices
including OpenBCI, Emotiv, NeuroSky, and clinical systems.
"""

import numpy as np
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
import warnings

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    warnings.warn("pyserial not available. Serial device support will be limited.")

try:
    import socket
    HAS_SOCKET = True
except ImportError:
    HAS_SOCKET = False


@dataclass
class DeviceConfig:
    """Configuration for brain signal acquisition devices."""
    
    device_type: str = 'openbci'
    sampling_rate: int = 256
    n_channels: int = 8
    port: str = '/dev/ttyUSB0'
    baudrate: int = 115200
    ip_address: str = '127.0.0.1'
    tcp_port: int = 12345
    buffer_size: int = 1024
    gain: float = 24.0
    impedance_check: bool = True


class BrainDevice(ABC):
    """Abstract base class for brain signal acquisition devices."""
    
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.is_connected = False
        self.is_streaming = False
        self.data_callback: Optional[Callable[[np.ndarray], None]] = None
        self.streaming_thread: Optional[threading.Thread] = None
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the device."""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Disconnect from the device."""
        pass
        
    @abstractmethod
    def start_streaming(self):
        """Start data streaming."""
        pass
        
    @abstractmethod
    def stop_streaming(self):
        """Stop data streaming."""
        pass
        
    @abstractmethod
    def read_data(self) -> Optional[np.ndarray]:
        """Read a batch of data from the device."""
        pass
        
    def set_data_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for new data."""
        self.data_callback = callback
        
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            'device_type': self.config.device_type,
            'sampling_rate': self.config.sampling_rate,
            'n_channels': self.config.n_channels,
            'is_connected': self.is_connected,
            'is_streaming': self.is_streaming
        }


class OpenBCIDevice(BrainDevice):
    """Driver for OpenBCI boards (Cyton, Ganglion, etc.)."""
    
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.serial_connection = None
        self.packet_buffer = bytearray()
        
    def connect(self) -> bool:
        """Connect to OpenBCI device via serial."""
        if not HAS_SERIAL:
            raise ImportError("pyserial required for OpenBCI connection")
            
        try:
            self.serial_connection = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=1.0
            )
            
            # Send reset command
            self.serial_connection.write(b'v')  # Reset command
            time.sleep(2.0)
            
            # Read board info
            response = self.serial_connection.read_all()
            if b'OpenBCI' in response:
                self.is_connected = True
                return True
            else:
                warnings.warn("OpenBCI board not responding properly")
                return False
                
        except Exception as e:
            warnings.warn(f"Failed to connect to OpenBCI: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from OpenBCI device."""
        if self.is_streaming:
            self.stop_streaming()
            
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write(b's')  # Stop streaming
            time.sleep(0.1)
            self.serial_connection.close()
            
        self.is_connected = False
        
    def start_streaming(self):
        """Start data streaming from OpenBCI."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")
            
        if self.is_streaming:
            return
            
        # Send start streaming command
        self.serial_connection.write(b'b')  # Start streaming
        
        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
    def stop_streaming(self):
        """Stop data streaming."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        
        if self.serial_connection:
            self.serial_connection.write(b's')  # Stop streaming
            
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
            
    def _streaming_loop(self):
        """Main streaming loop for OpenBCI data."""
        while self.is_streaming and self.serial_connection:
            try:
                # Read available data
                if self.serial_connection.in_waiting > 0:
                    new_data = self.serial_connection.read(self.serial_connection.in_waiting)
                    self.packet_buffer.extend(new_data)
                    
                    # Process complete packets
                    self._process_packets()
                    
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                warnings.warn(f"OpenBCI streaming error: {e}")
                break
                
    def _process_packets(self):
        """Process OpenBCI data packets."""
        # OpenBCI packet format: 33 bytes per packet
        # [start_byte][sample_num][ch1_msb][ch1_mid][ch1_lsb]...[aux1][aux2][aux3][end_byte]
        packet_size = 33
        
        while len(self.packet_buffer) >= packet_size:
            # Find packet start (0xA0)
            start_idx = self.packet_buffer.find(0xA0)
            
            if start_idx == -1:
                # No start byte found, clear buffer
                self.packet_buffer.clear()
                break
                
            if start_idx > 0:
                # Remove data before start byte
                self.packet_buffer = self.packet_buffer[start_idx:]
                
            if len(self.packet_buffer) < packet_size:
                break
                
            # Check end byte (0xC0)
            if self.packet_buffer[packet_size - 1] != 0xC0:
                # Invalid packet, remove start byte and continue
                self.packet_buffer = self.packet_buffer[1:]
                continue
                
            # Extract packet data
            packet = self.packet_buffer[:packet_size]
            self.packet_buffer = self.packet_buffer[packet_size:]
            
            # Parse EEG channels (8 channels, 3 bytes each, 24-bit signed)
            eeg_data = np.zeros(8)
            
            for i in range(8):
                offset = 2 + i * 3  # Skip start and sample number bytes
                
                # Combine 3 bytes into 24-bit signed integer
                value = (packet[offset] << 16) | (packet[offset + 1] << 8) | packet[offset + 2]
                
                # Convert to signed 24-bit
                if value >= 2**23:
                    value -= 2**24
                    
                # Convert to voltage (assuming default gain and reference)
                # OpenBCI uses 24-bit ADC with ±187.5mV range
                voltage = value * (4.5 / self.config.gain / (2**23))
                eeg_data[i] = voltage
                
            # Reshape for callback (channels, timepoints)
            eeg_data = eeg_data.reshape(-1, 1)
            
            # Call data callback if set
            if self.data_callback:
                self.data_callback(eeg_data)
                
    def read_data(self) -> Optional[np.ndarray]:
        """Read data synchronously (not recommended for real-time use)."""
        if not self.is_connected:
            return None
            
        # This is a simplified implementation
        # In practice, you'd want to use the streaming interface
        return None


class EmotivDevice(BrainDevice):
    """Driver for Emotiv EEG devices (EPOC, Insight, etc.)."""
    
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.cortex_session = None
        
    def connect(self) -> bool:
        """Connect to Emotiv device via Cortex API."""
        # This would require the Emotiv Cortex SDK
        # For now, this is a placeholder implementation
        warnings.warn("Emotiv Cortex SDK integration not implemented")
        return False
        
    def disconnect(self):
        """Disconnect from Emotiv device."""
        pass
        
    def start_streaming(self):
        """Start streaming from Emotiv device."""
        pass
        
    def stop_streaming(self):
        """Stop streaming."""
        pass
        
    def read_data(self) -> Optional[np.ndarray]:
        """Read data from Emotiv device."""
        return None


class LSLDevice(BrainDevice):
    """Driver for Lab Streaming Layer (LSL) devices."""
    
    def __init__(self, config: DeviceConfig, stream_name: str = 'EEG'):
        super().__init__(config)
        self.stream_name = stream_name
        self.inlet = None
        
    def connect(self) -> bool:
        """Connect to LSL stream."""
        try:
            import pylsl as lsl
            
            # Look for streams
            streams = lsl.resolve_stream('name', self.stream_name)
            
            if not streams:
                warnings.warn(f"No LSL stream found with name '{self.stream_name}'")
                return False
                
            # Create inlet
            self.inlet = lsl.StreamInlet(streams[0])
            self.is_connected = True
            return True
            
        except ImportError:
            warnings.warn("pylsl not available for LSL support")
            return False
        except Exception as e:
            warnings.warn(f"Failed to connect to LSL stream: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from LSL stream."""
        if self.is_streaming:
            self.stop_streaming()
            
        self.inlet = None
        self.is_connected = False
        
    def start_streaming(self):
        """Start streaming from LSL."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")
            
        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
    def stop_streaming(self):
        """Stop streaming."""
        self.is_streaming = False
        
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
            
    def _streaming_loop(self):
        """LSL streaming loop."""
        while self.is_streaming and self.inlet:
            try:
                # Pull samples (non-blocking)
                samples, timestamps = self.inlet.pull_chunk(timeout=0.1)
                
                if samples:
                    # Convert to numpy array and transpose
                    data = np.array(samples).T  # (channels, timepoints)
                    
                    if self.data_callback:
                        self.data_callback(data)
                        
            except Exception as e:
                warnings.warn(f"LSL streaming error: {e}")
                break
                
    def read_data(self) -> Optional[np.ndarray]:
        """Read data from LSL stream."""
        if not self.is_connected or not self.inlet:
            return None
            
        try:
            samples, timestamps = self.inlet.pull_chunk(timeout=1.0)
            if samples:
                return np.array(samples).T
            else:
                return None
        except Exception as e:
            warnings.warn(f"LSL read error: {e}")
            return None


class SimulatedDevice(BrainDevice):
    """Simulated brain device for testing and development."""
    
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.noise_level = 0.1
        self.signal_frequency = 10.0  # Hz
        self.time_offset = 0.0
        
    def connect(self) -> bool:
        """Connect to simulated device."""
        self.is_connected = True
        return True
        
    def disconnect(self):
        """Disconnect simulated device."""
        if self.is_streaming:
            self.stop_streaming()
        self.is_connected = False
        
    def start_streaming(self):
        """Start simulated streaming."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")
            
        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
    def stop_streaming(self):
        """Stop simulated streaming."""
        self.is_streaming = False
        
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
            
    def _streaming_loop(self):
        """Simulated streaming loop."""
        samples_per_chunk = max(1, self.config.sampling_rate // 10)  # 100ms chunks
        dt = 1.0 / self.config.sampling_rate
        
        while self.is_streaming:
            # Generate simulated EEG data
            t = np.arange(samples_per_chunk) * dt + self.time_offset
            
            # Create multi-channel sinusoidal signals with noise
            data = np.zeros((self.config.n_channels, samples_per_chunk))
            
            for ch in range(self.config.n_channels):
                # Different frequency for each channel
                freq = self.signal_frequency + ch * 2.0
                phase = ch * np.pi / 4  # Phase offset
                
                # Generate signal with noise
                signal = np.sin(2 * np.pi * freq * t + phase)
                noise = np.random.normal(0, self.noise_level, len(t))
                
                data[ch] = signal + noise
                
            # Scale to typical EEG amplitude range (microvolts)
            data *= 50.0  # 50 µV amplitude
            
            # Update time offset
            self.time_offset += samples_per_chunk * dt
            
            # Call data callback
            if self.data_callback:
                self.data_callback(data)
                
            # Sleep to maintain sampling rate
            time.sleep(samples_per_chunk / self.config.sampling_rate)
            
    def read_data(self) -> Optional[np.ndarray]:
        """Read simulated data."""
        if not self.is_connected:
            return None
            
        # Generate one sample of data
        t = np.array([self.time_offset])
        data = np.zeros((self.config.n_channels, 1))
        
        for ch in range(self.config.n_channels):
            freq = self.signal_frequency + ch * 2.0
            phase = ch * np.pi / 4
            signal = np.sin(2 * np.pi * freq * t + phase)
            noise = np.random.normal(0, self.noise_level, 1)
            data[ch] = (signal + noise) * 50.0
            
        self.time_offset += 1.0 / self.config.sampling_rate
        
        return data


def create_device(device_type: str, config: DeviceConfig) -> BrainDevice:
    """
    Factory function to create brain devices.
    
    Args:
        device_type: Type of device ('openbci', 'emotiv', 'lsl', 'simulated')
        config: Device configuration
        
    Returns:
        Brain device instance
    """
    device_type = device_type.lower()
    
    if device_type == 'openbci':
        return OpenBCIDevice(config)
    elif device_type == 'emotiv':
        return EmotivDevice(config)
    elif device_type == 'lsl':
        return LSLDevice(config)
    elif device_type == 'simulated':
        return SimulatedDevice(config)
    else:
        raise ValueError(f"Unknown device type: {device_type}")


class DeviceManager:
    """
    Manager for multiple brain devices.
    
    Provides unified interface for managing multiple devices
    and synchronizing data streams.
    """
    
    def __init__(self):
        self.devices: Dict[str, BrainDevice] = {}
        self.synchronized_callback: Optional[Callable[[Dict[str, np.ndarray]], None]] = None
        self.sync_buffer: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        self.sync_window = 0.1  # 100ms synchronization window
        
    def add_device(self, name: str, device: BrainDevice):
        """Add a device to the manager."""
        self.devices[name] = device
        self.sync_buffer[name] = []
        
        # Set up data callback for synchronization
        device.set_data_callback(lambda data, dev_name=name: self._device_callback(dev_name, data))
        
    def remove_device(self, name: str):
        """Remove a device from the manager."""
        if name in self.devices:
            self.devices[name].disconnect()
            del self.devices[name]
            del self.sync_buffer[name]
            
    def connect_all(self) -> Dict[str, bool]:
        """Connect all devices."""
        results = {}
        for name, device in self.devices.items():
            results[name] = device.connect()
        return results
        
    def disconnect_all(self):
        """Disconnect all devices."""
        for device in self.devices.values():
            device.disconnect()
            
    def start_all_streaming(self):
        """Start streaming on all devices."""
        for device in self.devices.values():
            if device.is_connected:
                device.start_streaming()
                
    def stop_all_streaming(self):
        """Stop streaming on all devices."""
        for device in self.devices.values():
            device.stop_streaming()
            
    def set_synchronized_callback(self, callback: Callable[[Dict[str, np.ndarray]], None]):
        """Set callback for synchronized data from all devices."""
        self.synchronized_callback = callback
        
    def _device_callback(self, device_name: str, data: np.ndarray):
        """Handle data from individual devices."""
        timestamp = time.time()
        self.sync_buffer[device_name].append((timestamp, data))
        
        # Clean old data
        cutoff_time = timestamp - self.sync_window
        self.sync_buffer[device_name] = [
            (t, d) for t, d in self.sync_buffer[device_name] 
            if t > cutoff_time
        ]
        
        # Check if we can synchronize
        self._check_synchronization()
        
    def _check_synchronization(self):
        """Check if synchronized data is available from all devices."""
        if not self.synchronized_callback:
            return
            
        current_time = time.time()
        sync_data = {}
        
        # Find data from all devices within sync window
        for device_name, buffer in self.sync_buffer.items():
            if not buffer:
                return  # No data from this device
                
            # Find most recent data within sync window
            recent_data = None
            for timestamp, data in reversed(buffer):
                if current_time - timestamp <= self.sync_window:
                    recent_data = data
                    break
                    
            if recent_data is None:
                return  # No recent data from this device
                
            sync_data[device_name] = recent_data
            
        # Call synchronized callback
        if len(sync_data) == len(self.devices):
            self.synchronized_callback(sync_data)
            
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all devices."""
        return {
            name: device.get_device_info() 
            for name, device in self.devices.items()
        }