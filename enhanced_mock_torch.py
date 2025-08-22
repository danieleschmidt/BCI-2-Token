"""
Enhanced comprehensive mock PyTorch implementation for full BCI-2-Token testing.
Provides complete interface compatibility for all PyTorch features used in the codebase.
"""

import warnings
import numpy as np
import sys

class MockTensor:
    def __init__(self, data, requires_grad=False, device='cpu', dtype=None):
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, (int, float)):
            data = np.array([data])
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = data.shape if hasattr(data, 'shape') else ()
        self.device = MockDevice(device)
        self.dtype = dtype
        
    def __repr__(self):
        return f"MockTensor({self.data})"
        
    def __getitem__(self, key):
        return MockTensor(self.data[key], self.requires_grad, str(self.device))
        
    def __setitem__(self, key, value):
        if isinstance(value, MockTensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
        
    def __len__(self):
        return len(self.data)
        
    def numpy(self):
        return self.data
        
    def detach(self):
        return MockTensor(self.data.copy(), requires_grad=False, device=str(self.device))
        
    def backward(self, gradient=None):
        pass
        
    def item(self):
        return float(self.data) if np.isscalar(self.data) else self.data.item()
        
    def to(self, device):
        return MockTensor(self.data, self.requires_grad, str(device))
        
    def cpu(self):
        return self.to('cpu')
        
    def cuda(self):
        return self.to('cuda')
        
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim] if dim < len(self.shape) else 1
        
    def dim(self):
        return len(self.shape)
        
    def view(self, *shape):
        return MockTensor(self.data.reshape(shape), self.requires_grad, str(self.device))
        
    def reshape(self, *shape):
        return self.view(*shape)
        
    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self.data, dim), self.requires_grad, str(self.device))
        
    def squeeze(self, dim=None):
        return MockTensor(np.squeeze(self.data, dim), self.requires_grad, str(self.device))
        
    def transpose(self, dim0, dim1):
        axes = list(range(len(self.shape)))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return MockTensor(np.transpose(self.data, axes), self.requires_grad, str(self.device))
        
    def permute(self, *dims):
        return MockTensor(np.transpose(self.data, dims), self.requires_grad, str(self.device))
        
    def mean(self, dim=None, keepdim=False):
        return MockTensor(np.mean(self.data, axis=dim, keepdims=keepdim))
        
    def sum(self, dim=None, keepdim=False):
        return MockTensor(np.sum(self.data, axis=dim, keepdims=keepdim))
        
    def max(self, dim=None, keepdim=False):
        if dim is None:
            return MockTensor(np.max(self.data))
        result = np.max(self.data, axis=dim, keepdims=keepdim)
        indices = np.argmax(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result), MockTensor(indices)
        
    def min(self, dim=None, keepdim=False):
        if dim is None:
            return MockTensor(np.min(self.data))
        result = np.min(self.data, axis=dim, keepdims=keepdim)
        indices = np.argmin(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result), MockTensor(indices)
        
    def clamp(self, min_val=None, max_val=None):
        return MockTensor(np.clip(self.data, min_val, max_val))
        
    def float(self):
        return MockTensor(self.data.astype(np.float32), self.requires_grad, str(self.device))
        
    def long(self):
        return MockTensor(self.data.astype(np.int64), self.requires_grad, str(self.device))
        
    def bool(self):
        return MockTensor(self.data.astype(bool), self.requires_grad, str(self.device))
        
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data * other.data, self.requires_grad or other.requires_grad, str(self.device))
        else:
            return MockTensor(self.data * other, self.requires_grad, str(self.device))
            
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data, self.requires_grad or other.requires_grad, str(self.device))
        else:
            return MockTensor(self.data + other, self.requires_grad, str(self.device))
            
    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data - other.data, self.requires_grad or other.requires_grad, str(self.device))
        else:
            return MockTensor(self.data - other, self.requires_grad, str(self.device))
            
    def __rsub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(other.data - self.data, self.requires_grad or other.requires_grad, str(self.device))
        else:
            return MockTensor(other - self.data, self.requires_grad, str(self.device))
            
    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data / other.data, self.requires_grad or other.requires_grad, str(self.device))
        else:
            return MockTensor(self.data / other, self.requires_grad, str(self.device))
            
    def __rtruediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(other.data / self.data, self.requires_grad or other.requires_grad, str(self.device))
        else:
            return MockTensor(other / self.data, self.requires_grad, str(self.device))

def tensor(data, requires_grad=False, device='cpu', dtype=None):
    """Create a mock tensor"""
    return MockTensor(data, requires_grad, device, dtype)

class MockDevice:
    def __init__(self, device_type='cpu'):
        self.type = device_type
        
    def __str__(self):
        return self.type
        
    def __repr__(self):
        return f"device(type='{self.type}')"

def device(device_str='cpu'):
    return MockDevice(device_str)

class MockParameter(MockTensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad)

# Mock nn module
class MockModule:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
        
    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            for param in module.parameters():
                yield param
                
    def named_parameters(self):
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param
        
    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
        
    def eval(self):
        return self.train(False)
        
    def to(self, device):
        return self
        
    def cuda(self):
        return self.to('cuda')
        
    def cpu(self):
        return self.to('cpu')
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def state_dict(self):
        return {}
        
    def load_state_dict(self, state_dict, strict=True):
        pass
        
    def zero_grad(self):
        pass
        
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

class MockLinear(MockModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = MockParameter(np.random.randn(out_features, in_features) * 0.1)
        self.bias = MockParameter(np.random.randn(out_features) * 0.1) if bias else None
        self._parameters['weight'] = self.weight
        if bias:
            self._parameters['bias'] = self.bias
        
    def forward(self, x):
        if isinstance(x, MockTensor):
            output = np.dot(x.data, self.weight.data.T)
            if self.bias is not None:
                output = output + self.bias.data
            return MockTensor(output)
        return x

class MockEmbedding(MockModule):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = MockParameter(np.random.randn(num_embeddings, embedding_dim) * 0.1)
        self._parameters['weight'] = self.weight
        
    def forward(self, input):
        if isinstance(input, MockTensor):
            # Simple embedding lookup
            indices = input.data.astype(int)
            output = self.weight.data[indices]
            return MockTensor(output)
        return input

class MockConv1d(MockModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = MockParameter(np.random.randn(out_channels, in_channels, kernel_size) * 0.1)
        self.bias = MockParameter(np.random.randn(out_channels) * 0.1) if bias else None
        self._parameters['weight'] = self.weight
        if bias:
            self._parameters['bias'] = self.bias
            
    def forward(self, x):
        # Simplified conv1d
        if isinstance(x, MockTensor):
            batch_size, in_channels, length = x.data.shape
            out_length = length  # Simplified
            output = np.random.randn(batch_size, self.out_channels, out_length)
            return MockTensor(output)
        return x

class MockBatchNorm1d(MockModule):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = MockParameter(np.ones(num_features))
        self.bias = MockParameter(np.zeros(num_features))
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias
        
    def forward(self, x):
        return x

class MockLayerNorm(MockModule):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.weight = MockParameter(np.ones(normalized_shape))
        self.bias = MockParameter(np.zeros(normalized_shape))
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias
        
    def forward(self, x):
        return x

class MockDropout(MockModule):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        return x

class MockReLU(MockModule):
    def __init__(self, inplace=False):
        super().__init__()
        
    def forward(self, x):
        if isinstance(x, MockTensor):
            return MockTensor(np.maximum(0, x.data))
        return x

class MockGELU(MockModule):
    def forward(self, x):
        if isinstance(x, MockTensor):
            data = x.data
            result = 0.5 * data * (1 + np.tanh(np.sqrt(2/np.pi) * (data + 0.044715 * data**3)))
            return MockTensor(result)
        return x

class MockSequential(MockModule):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = modules
        
    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

class MockLoss(MockModule):
    def forward(self, input, target):
        return MockTensor(np.array(0.5))

class MockTransformer(MockModule):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, src, tgt=None):
        if isinstance(src, MockTensor):
            batch_size, seq_len, d_model = src.data.shape
            return MockTensor(np.random.randn(batch_size, seq_len, d_model))
        return src

class MockTransformerEncoderLayer(MockModule):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False, norm_first=False, **kwargs):
        super().__init__()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        return src

class MockTransformerEncoder(MockModule):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        return src

class MockTransformerDecoderLayer(MockModule):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False, norm_first=False, **kwargs):
        super().__init__()
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return tgt

class MockTransformerDecoder(MockModule):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return tgt

class MockCTCLoss(MockModule):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super().__init__()
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return MockTensor(np.array(1.0))

class nn:
    Module = MockModule
    Linear = MockLinear
    Embedding = MockEmbedding
    Conv1d = MockConv1d
    BatchNorm1d = MockBatchNorm1d
    LayerNorm = MockLayerNorm
    Dropout = MockDropout
    ReLU = MockReLU
    GELU = MockGELU
    Transformer = MockTransformer
    TransformerEncoder = MockTransformerEncoder
    TransformerEncoderLayer = MockTransformerEncoderLayer
    TransformerDecoder = MockTransformerDecoder
    TransformerDecoderLayer = MockTransformerDecoderLayer
    Parameter = MockParameter
    Sequential = MockSequential
    
    @staticmethod
    def CrossEntropyLoss(ignore_index=-100):
        return MockLoss()
        
    @staticmethod
    def MSELoss():
        return MockLoss()
        
    @staticmethod
    def L1Loss():
        return MockLoss()
        
    @staticmethod
    def BCELoss():
        return MockLoss()
        
    @staticmethod
    def CTCLoss(blank=0, reduction='mean', zero_infinity=False):
        return MockCTCLoss(blank, reduction, zero_infinity)

class MockOptimizer:
    def __init__(self, parameters, lr=0.001, **kwargs):
        self.param_groups = [{'params': list(parameters), 'lr': lr, **kwargs}]
        
    def step(self):
        pass
        
    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad = None

class MockScheduler:
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        
    def step(self):
        pass
        
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class optim:
    @staticmethod
    def Adam(parameters, lr=0.001, **kwargs):
        return MockOptimizer(parameters, lr, **kwargs)
        
    @staticmethod
    def SGD(parameters, lr=0.01, **kwargs):
        return MockOptimizer(parameters, lr, **kwargs)
        
    @staticmethod
    def AdamW(parameters, lr=0.001, **kwargs):
        return MockOptimizer(parameters, lr, **kwargs)
        
    class lr_scheduler:
        @staticmethod
        def StepLR(optimizer, step_size, gamma=0.1):
            return MockScheduler(optimizer)
            
        @staticmethod
        def ExponentialLR(optimizer, gamma):
            return MockScheduler(optimizer)
            
        @staticmethod
        def CosineAnnealingLR(optimizer, T_max):
            return MockScheduler(optimizer)

# Mock functional
class functional:
    @staticmethod
    def relu(x, inplace=False):
        if isinstance(x, MockTensor):
            return MockTensor(np.maximum(0, x.data))
        return x
        
    @staticmethod
    def gelu(x):
        if isinstance(x, MockTensor):
            data = x.data
            result = 0.5 * data * (1 + np.tanh(np.sqrt(2/np.pi) * (data + 0.044715 * data**3)))
            return MockTensor(result)
        return x
        
    @staticmethod
    def softmax(x, dim=-1):
        if isinstance(x, MockTensor):
            exp_x = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
            return MockTensor(exp_x / np.sum(exp_x, axis=dim, keepdims=True))
        return x
        
    @staticmethod
    def log_softmax(x, dim=-1):
        if isinstance(x, MockTensor):
            softmax_x = functional.softmax(x, dim)
            return MockTensor(np.log(softmax_x.data + 1e-8))
        return x
        
    @staticmethod
    def dropout(x, p=0.5, training=True, inplace=False):
        if training and isinstance(x, MockTensor):
            mask = np.random.random(x.data.shape) > p
            return MockTensor(x.data * mask / (1 - p))
        return x
        
    @staticmethod
    def cross_entropy(input, target, ignore_index=-100):
        return MockTensor(np.array(1.0))
        
    @staticmethod
    def mse_loss(input, target):
        return MockTensor(np.array(0.5))
        
    @staticmethod
    def interpolate(input, size=None, scale_factor=None, mode='nearest'):
        return input
        
    @staticmethod
    def pad(input, pad, mode='constant', value=0):
        return input

def randn(*shape, requires_grad=False, device='cpu', dtype=None):
    return MockTensor(np.random.randn(*shape), requires_grad, device, dtype)

def rand(*shape, requires_grad=False, device='cpu', dtype=None):
    return MockTensor(np.random.rand(*shape), requires_grad, device, dtype)

def zeros(*shape, requires_grad=False, device='cpu', dtype=None):
    return MockTensor(np.zeros(shape), requires_grad, device, dtype)

def ones(*shape, requires_grad=False, device='cpu', dtype=None):
    return MockTensor(np.ones(shape), requires_grad, device, dtype)

def eye(n, m=None, requires_grad=False, device='cpu', dtype=None):
    if m is None:
        m = n
    return MockTensor(np.eye(n, m), requires_grad, device, dtype)

def arange(start, end=None, step=1, requires_grad=False, device='cpu', dtype=None):
    if end is None:
        end = start
        start = 0
    return MockTensor(np.arange(start, end, step), requires_grad, device, dtype)

def linspace(start, end, steps, requires_grad=False, device='cpu', dtype=None):
    return MockTensor(np.linspace(start, end, steps), requires_grad, device, dtype)

def exp(input):
    if isinstance(input, MockTensor):
        return MockTensor(np.exp(input.data))
    return MockTensor(np.exp(input))

def sin(input):
    if isinstance(input, MockTensor):
        return MockTensor(np.sin(input.data))
    return MockTensor(np.sin(input))

def cos(input):
    if isinstance(input, MockTensor):
        return MockTensor(np.cos(input.data))
    return MockTensor(np.cos(input))

def sqrt(input):
    if isinstance(input, MockTensor):
        return MockTensor(np.sqrt(input.data))
    return MockTensor(np.sqrt(input))

def cumprod(input, dim):
    if isinstance(input, MockTensor):
        return MockTensor(np.cumprod(input.data, axis=dim))
    return MockTensor(np.cumprod(input, axis=dim))

def randint(low, high, size, requires_grad=False, device='cpu', dtype=None):
    return MockTensor(np.random.randint(low, high, size), requires_grad, device, dtype)

def full(size, fill_value, requires_grad=False, device='cpu', dtype=None):
    return MockTensor(np.full(size, fill_value), requires_grad, device, dtype)

def cat(tensors, dim=0):
    arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
    return MockTensor(np.concatenate(arrays, axis=dim))

def stack(tensors, dim=0):
    arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
    return MockTensor(np.stack(arrays, axis=dim))

def save(obj, path):
    warnings.warn("torch.save called with mock implementation")
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load(path, map_location=None):
    warnings.warn("torch.load called with mock implementation")
    try:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return {}

def manual_seed(seed):
    np.random.seed(seed)

class cuda:
    @staticmethod
    def is_available():
        return False
        
    @staticmethod
    def device_count():
        return 0
        
    @staticmethod
    def current_device():
        return 0
        
    @staticmethod
    def get_device_name(device=None):
        return "Mock CUDA Device"

class autograd:
    @staticmethod
    def grad(outputs, inputs, create_graph=False, retain_graph=False):
        return [MockTensor(np.ones_like(inp.data)) for inp in inputs]

def no_grad():
    class NoGradContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoGradContext()

def enable_grad():
    class EnableGradContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return EnableGradContext()

# Set up module structure
Tensor = MockTensor
FloatTensor = MockTensor
LongTensor = MockTensor
BoolTensor = MockTensor

# Add dtype support
float = np.float32
long = np.int64
bool = np.bool_

# Mock torch.utils for DataLoader and other utilities
class MockDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i+self.batch_size]
            yield batch
            
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class MockTensorDataset:
    def __init__(self, *tensors):
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
        
    def __len__(self):
        return len(self.tensors[0])

class utils:
    class data:
        DataLoader = MockDataLoader
        TensorDataset = MockTensorDataset
        
        @staticmethod
        def random_split(dataset, lengths):
            # Simple mock split
            return [dataset[:l] for l in lengths]

# Install mocks in sys.modules
sys.modules['torch'] = sys.modules[__name__]
sys.modules['torch.nn'] = nn
sys.modules['torch.optim'] = optim
sys.modules['torch.nn.functional'] = functional
sys.modules['torch.cuda'] = cuda
sys.modules['torch.autograd'] = autograd
sys.modules['torch.utils'] = utils
sys.modules['torch.utils.data'] = utils.data

# Commonly used imports
F = functional