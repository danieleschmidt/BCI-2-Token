"""
Minimal mock PyTorch implementation for testing BCI-2-Token without full PyTorch.
Only implements the interfaces needed for basic functionality testing.
"""

import numpy as np


class Tensor:
    """Mock torch.Tensor for testing."""
    
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data)
        else:
            self.data = np.array([data])
            
    def __getitem__(self, key):
        return Tensor(self.data[key])
        
    def shape(self):
        return self.data.shape
        
    def size(self, dim=None):
        if dim is None:
            return self.data.shape
        return self.data.shape[dim]
        
    def numpy(self):
        return self.data
        
    def tolist(self):
        return self.data.tolist()
        
    def item(self):
        return self.data.item()
        
    def transpose(self, dim0, dim1):
        return Tensor(np.transpose(self.data, (dim0, dim1)))
        
    def to(self, device):
        return self  # Ignore device for mock
        
    def eval(self):
        return self
        
    def train(self):
        return self


class Module:
    """Mock torch.nn.Module for testing."""
    
    def __init__(self):
        pass
        
    def eval(self):
        return self
        
    def train(self):
        return self
        
    def to(self, device):
        return self
        
    def parameters(self):
        return []
        
    def state_dict(self):
        return {}
        
    def load_state_dict(self, state_dict):
        pass


class MockDevice:
    """Mock device for testing."""
    def __init__(self, device_str):
        self.device_str = device_str


# Mock torch module structure
class torch:
    Tensor = Tensor
    FloatTensor = Tensor
    LongTensor = Tensor
    
    @staticmethod
    def tensor(data, dtype=None):
        return Tensor(data)
        
    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape))
        
    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape))
        
    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape))
        
    @staticmethod
    def rand(*shape):
        return Tensor(np.random.rand(*shape))
        
    @staticmethod
    def save(obj, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
            
    @staticmethod
    def load(path, map_location=None):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    @staticmethod
    def device(device_str):
        return MockDevice(device_str)
        
    class cuda:
        @staticmethod
        def is_available():
            return False
            
        @staticmethod
        def device_count():
            return 0
    
    class nn:
        Module = Module
        
        class Linear(Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = Tensor(np.random.randn(out_features, in_features))
                self.bias = Tensor(np.random.randn(out_features))
                
        class Conv1d(Module):
            def __init__(self, in_channels, out_channels, kernel_size, padding=0):
                super().__init__()
                
        class Embedding(Module):
            def __init__(self, num_embeddings, embedding_dim):
                super().__init__()


# Mock F module
class F:
    @staticmethod
    def cross_entropy(input, target, ignore_index=-100):
        return Tensor(np.array(1.0))  # Mock loss value
        
    @staticmethod
    def log_softmax(input, dim=-1):
        # Simple mock log softmax
        return Tensor(np.log(np.random.rand(*input.data.shape)))


# Replace imports in modules that need torch
import sys
sys.modules['torch'] = torch
sys.modules['torch.nn'] = torch.nn  
sys.modules['torch.nn.functional'] = F