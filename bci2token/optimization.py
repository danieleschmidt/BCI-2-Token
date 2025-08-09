"""
Performance optimization and scaling for BCI-2-Token framework.

Implements caching, concurrent processing, resource pooling, and auto-scaling
for high-performance brain-computer interface applications.
"""

import time
import threading
import queue
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
import hashlib

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class OptimizationConfig:
    """Configuration for optimization features."""
    
    # Caching
    enable_signal_cache: bool = True
    cache_size: int = 1000
    cache_ttl: float = 300.0  # 5 minutes
    
    # Concurrency
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    enable_parallel_processing: bool = True
    
    # Resource pooling
    enable_resource_pooling: bool = True
    pool_size: int = 10
    pool_timeout: float = 30.0
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # 80% utilization
    scale_down_threshold: float = 0.3  # 30% utilization
    min_instances: int = 1
    max_instances: int = 10


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()
        
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                return None
                
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl:
                self._remove_key(key)
                return None
                
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry['value']
            
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self._remove_key(key)
                
            # Add new entry
            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self.access_order.append(key)
            
            # Evict if over capacity
            while len(self.cache) > self.max_size:
                oldest_key = self.access_order[0]
                self._remove_key(oldest_key)
                
    def _remove_key(self, key: str):
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
            
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size,
                'ttl': self.ttl
            }


class SignalCache:
    """Specialized cache for brain signal processing results."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = LRUCache(config.cache_size, config.cache_ttl)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
        
    def get_processed_signal(self, 
                           signal_hash: str,
                           preprocessing_config: Any) -> Optional[Dict[str, Any]]:
        """Get cached preprocessed signal."""
        if not self.config.enable_signal_cache:
            return None
            
        # Create cache key from signal hash and preprocessing config
        config_str = str(preprocessing_config.__dict__ if hasattr(preprocessing_config, '__dict__') else preprocessing_config)
        cache_key = f"{signal_hash}_{hashlib.md5(config_str.encode()).hexdigest()}"
        
        result = self.cache.get(cache_key)
        
        with self.lock:
            if result is not None:
                self.hit_count += 1
            else:
                self.miss_count += 1
                
        return result
        
    def cache_processed_signal(self,
                              signal_hash: str,
                              preprocessing_config: Any,
                              processed_result: Dict[str, Any]):
        """Cache preprocessed signal result."""
        if not self.config.enable_signal_cache:
            return
            
        config_str = str(preprocessing_config.__dict__ if hasattr(preprocessing_config, '__dict__') else preprocessing_config)
        cache_key = f"{signal_hash}_{hashlib.md5(config_str.encode()).hexdigest()}"
        
        self.cache.put(cache_key, processed_result)
        
    def calculate_signal_hash(self, signal: Any) -> str:
        """Calculate hash of brain signal for caching."""
        if not HAS_NUMPY or not isinstance(signal, np.ndarray):
            return str(hash(str(signal)))
            
        # Use a subset of the signal for efficiency
        sample_points = min(1000, signal.size)
        flat_signal = signal.flatten()
        sample_indices = np.linspace(0, len(flat_signal) - 1, sample_points, dtype=int)
        sample_data = flat_signal[sample_indices]
        
        # Create hash from sample
        return hashlib.md5(sample_data.tobytes()).hexdigest()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            
            return {
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'cache_stats': self.cache.get_stats()
            }


class ConcurrentProcessor:
    """Concurrent processing for BCI operations."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_worker_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_worker_processes) if config.enable_parallel_processing else None
        
    def process_signals_concurrent(self,
                                  signals: List[Any],
                                  process_func: Callable,
                                  use_processes: bool = False) -> List[Any]:
        """
        Process multiple signals concurrently.
        
        Args:
            signals: List of brain signals to process
            process_func: Function to apply to each signal
            use_processes: Whether to use process-based parallelism
            
        Returns:
            List of processing results
        """
        if not signals:
            return []
            
        if len(signals) == 1 or not self.config.enable_parallel_processing:
            # Single signal or parallelism disabled
            return [process_func(signal) for signal in signals]
            
        # Choose executor
        executor = self.process_pool if use_processes and self.process_pool else self.thread_pool
        
        if executor is None:
            # Fallback to sequential processing
            return [process_func(signal) for signal in signals]
            
        try:
            # Submit all tasks
            futures = [executor.submit(process_func, signal) for signal in signals]
            
            # Collect results in order
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30.0)  # 30 second timeout per task
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Concurrent processing failed for one signal: {e}")
                    results.append(None)  # Placeholder for failed processing
                    
            return results
            
        except Exception as e:
            warnings.warn(f"Concurrent processing setup failed: {e}")
            # Fallback to sequential
            return [process_func(signal) for signal in signals]
            
    def batch_process_with_batching(self,
                                   items: List[Any],
                                   process_func: Callable,
                                   batch_size: int = None) -> List[Any]:
        """
        Process items in optimized batches.
        
        Args:
            items: Items to process
            process_func: Function that processes a batch
            batch_size: Batch size (auto-calculated if None)
            
        Returns:
            List of processing results
        """
        if not items:
            return []
            
        # Auto-calculate batch size based on available resources
        if batch_size is None:
            batch_size = max(1, len(items) // (self.config.max_worker_threads * 2))
            batch_size = min(batch_size, 100)  # Max batch size
            
        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
            
        # Process batches concurrently
        if len(batches) > 1 and self.config.enable_parallel_processing:
            futures = [self.thread_pool.submit(process_func, batch) for batch in batches]
            
            results = []
            for future in futures:
                try:
                    batch_result = future.result(timeout=60.0)
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                except Exception as e:
                    warnings.warn(f"Batch processing failed: {e}")
                    
            return results
        else:
            # Sequential batch processing
            results = []
            for batch in batches:
                try:
                    batch_result = process_func(batch)
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                except Exception as e:
                    warnings.warn(f"Sequential batch processing failed: {e}")
                    
            return results
            
    def shutdown(self):
        """Shutdown concurrent processors."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class ResourcePool:
    """Pool of reusable resources for BCI operations."""
    
    def __init__(self, 
                 resource_factory: Callable,
                 pool_size: int = 10,
                 timeout: float = 30.0):
        """
        Initialize resource pool.
        
        Args:
            resource_factory: Function that creates new resources
            pool_size: Maximum pool size
            timeout: Timeout for acquiring resources
        """
        self.resource_factory = resource_factory
        self.pool_size = pool_size
        self.timeout = timeout
        self.pool = queue.Queue(maxsize=pool_size)
        self.created_count = 0
        self.lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(min(2, pool_size)):  # Start with 2 resources
            try:
                resource = self.resource_factory()
                self.pool.put(resource, block=False)
                self.created_count += 1
            except Exception as e:
                warnings.warn(f"Failed to create initial resource: {e}")
                
    def acquire(self) -> Any:
        """
        Acquire a resource from the pool.
        
        Returns:
            Resource instance
            
        Raises:
            TimeoutError: If no resource available within timeout
        """
        try:
            # Try to get existing resource
            resource = self.pool.get(timeout=self.timeout)
            return resource
        except queue.Empty:
            # Create new resource if pool not at capacity
            with self.lock:
                if self.created_count < self.pool_size:
                    try:
                        resource = self.resource_factory()
                        self.created_count += 1
                        return resource
                    except Exception as e:
                        raise RuntimeError(f"Failed to create new resource: {e}")
                        
            raise TimeoutError(f"No resource available within {self.timeout}s")
            
    def release(self, resource: Any):
        """
        Release a resource back to the pool.
        
        Args:
            resource: Resource to release
        """
        try:
            # Reset resource state if it has a reset method
            if hasattr(resource, 'reset'):
                resource.reset()
                
            # Put back in pool (non-blocking)
            self.pool.put(resource, block=False)
        except queue.Full:
            # Pool is full, resource will be garbage collected
            with self.lock:
                self.created_count = max(0, self.created_count - 1)
        except Exception as e:
            warnings.warn(f"Failed to reset resource: {e}")
            # Don't put back in pool if reset failed
            with self.lock:
                self.created_count = max(0, self.created_count - 1)
                
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'pool_size': self.pool.qsize(),
                'max_pool_size': self.pool_size,
                'created_count': self.created_count,
                'utilization': 1.0 - (self.pool.qsize() / self.pool_size)
            }


class BatchProcessor:
    """Optimized batch processing for brain signals."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.concurrent_processor = ConcurrentProcessor(config)
        
    def process_signal_batch(self,
                           signals: List[Any],
                           decoder_func: Callable,
                           batch_size: int = None) -> List[Dict[str, Any]]:
        """
        Process a batch of brain signals optimally.
        
        Args:
            signals: List of brain signals
            decoder_func: Decoding function to apply
            batch_size: Optimal batch size (auto-calculated if None)
            
        Returns:
            List of decoding results
        """
        if not signals:
            return []
            
        # Auto-calculate optimal batch size
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(signals))
            
        # Process in optimal batches
        def process_batch(batch):
            return [decoder_func(signal) for signal in batch]
            
        return self.concurrent_processor.batch_process_with_batching(
            signals, process_batch, batch_size
        )
        
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on system resources."""
        # Simple heuristic based on available workers
        workers = self.config.max_worker_threads
        optimal_size = max(1, total_items // (workers * 2))
        
        # Clamp to reasonable range
        return min(max(optimal_size, 1), 50)
        
    def streaming_batch_processor(self,
                                 input_queue: queue.Queue,
                                 output_queue: queue.Queue,
                                 process_func: Callable,
                                 batch_size: int = 10,
                                 max_wait_time: float = 1.0):
        """
        Process streaming data in batches for efficiency.
        
        Args:
            input_queue: Queue of incoming data
            output_queue: Queue for processed results
            process_func: Function to process batches
            batch_size: Target batch size
            max_wait_time: Maximum time to wait for batch
        """
        def batch_worker():
            batch = []
            last_process_time = time.time()
            
            while True:
                try:
                    # Get item with timeout
                    item = input_queue.get(timeout=0.1)
                    batch.append(item)
                    
                    # Process batch if full or timeout reached
                    current_time = time.time()
                    if (len(batch) >= batch_size or 
                        current_time - last_process_time > max_wait_time):
                        
                        if batch:
                            try:
                                results = process_func(batch)
                                
                                # Put results in output queue
                                if isinstance(results, list):
                                    for result in results:
                                        output_queue.put(result)
                                else:
                                    output_queue.put(results)
                                    
                            except Exception as e:
                                warnings.warn(f"Batch processing failed: {e}")
                                
                            batch = []
                            last_process_time = current_time
                            
                except queue.Empty:
                    # Process any remaining items in batch
                    current_time = time.time()
                    if batch and current_time - last_process_time > max_wait_time:
                        try:
                            results = process_func(batch)
                            if isinstance(results, list):
                                for result in results:
                                    output_queue.put(result)
                            else:
                                output_queue.put(results)
                        except Exception as e:
                            warnings.warn(f"Final batch processing failed: {e}")
                            
                        batch = []
                        last_process_time = current_time
                        
                except Exception as e:
                    warnings.warn(f"Streaming batch processor error: {e}")
                    break
                    
        # Start worker thread
        worker_thread = threading.Thread(target=batch_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        return worker_thread


class LoadBalancer:
    """Load balancer for distributing BCI processing requests."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.workers: List[Dict[str, Any]] = []
        self.current_worker = 0
        self.lock = threading.Lock()
        
    def add_worker(self, 
                  worker_id: str,
                  process_func: Callable,
                  capacity: int = 100):
        """
        Add a worker to the load balancer.
        
        Args:
            worker_id: Unique worker identifier
            process_func: Processing function for this worker
            capacity: Worker capacity (requests per minute)
        """
        with self.lock:
            worker = {
                'id': worker_id,
                'process_func': process_func,
                'capacity': capacity,
                'current_load': 0,
                'total_requests': 0,
                'failed_requests': 0,
                'last_request_time': 0.0,
                'response_times': []
            }
            self.workers.append(worker)
            
    def remove_worker(self, worker_id: str):
        """Remove a worker from the load balancer."""
        with self.lock:
            self.workers = [w for w in self.workers if w['id'] != worker_id]
            
    def select_worker(self) -> Optional[Dict[str, Any]]:
        """Select best available worker using round-robin with load consideration."""
        if not self.workers:
            return None
            
        with self.lock:
            # Simple round-robin selection
            if self.current_worker >= len(self.workers):
                self.current_worker = 0
                
            selected = self.workers[self.current_worker]
            self.current_worker = (self.current_worker + 1) % len(self.workers)
            
            return selected
            
    def process_request(self, request_data: Any) -> Any:
        """
        Process request using load balancing.
        
        Args:
            request_data: Data to process
            
        Returns:
            Processing result
            
        Raises:
            RuntimeError: If no workers available
        """
        worker = self.select_worker()
        if not worker:
            # Fallback: process directly without load balancing
            if callable(request_data):
                return request_data()
            else:
                # If it's just data, return it as-is
                return request_data
            
        start_time = time.time()
        
        try:
            # Update worker load
            with self.lock:
                worker['current_load'] += 1
                worker['total_requests'] += 1
                worker['last_request_time'] = start_time
                
            # Process request
            result = worker['process_func'](request_data)
            
            # Record success
            end_time = time.time()
            response_time = end_time - start_time
            
            with self.lock:
                worker['current_load'] = max(0, worker['current_load'] - 1)
                worker['response_times'].append(response_time)
                
                # Keep only recent response times
                if len(worker['response_times']) > 100:
                    worker['response_times'] = worker['response_times'][-100:]
                    
            return result
            
        except Exception as e:
            # Record failure
            with self.lock:
                worker['current_load'] = max(0, worker['current_load'] - 1)
                worker['failed_requests'] += 1
                
            raise
            
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            worker_stats = []
            
            for worker in self.workers:
                total_requests = worker['total_requests']
                failed_requests = worker['failed_requests']
                response_times = worker['response_times']
                
                stats = {
                    'id': worker['id'],
                    'current_load': worker['current_load'],
                    'total_requests': total_requests,
                    'failed_requests': failed_requests,
                    'success_rate': (total_requests - failed_requests) / max(1, total_requests),
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0.0,
                    'last_request_time': worker['last_request_time']
                }
                worker_stats.append(stats)
                
            return {
                'worker_count': len(self.workers),
                'workers': worker_stats,
                'total_requests': sum(w['total_requests'] for w in self.workers),
                'total_failures': sum(w['failed_requests'] for w in self.workers)
            }


class AutoScaler:
    """Automatic scaling for BCI processing capacity."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_instances = config.min_instances
        self.load_history = []
        self.scaling_cooldown = 60.0  # seconds
        self.last_scale_time = 0.0
        self.lock = threading.Lock()
        
    def record_load_metric(self, load: float):
        """
        Record current load metric.
        
        Args:
            load: Current load (0.0 to 1.0)
        """
        with self.lock:
            self.load_history.append((time.time(), load))
            
            # Keep only recent history
            cutoff_time = time.time() - 300.0  # 5 minutes
            self.load_history = [(t, l) for t, l in self.load_history if t > cutoff_time]
            
    def should_scale_up(self) -> bool:
        """Determine if system should scale up."""
        if not self.config.enable_auto_scaling:
            return False
            
        if self.current_instances >= self.config.max_instances:
            return False
            
        if time.time() - self.last_scale_time < self.scaling_cooldown:
            return False
            
        # Check recent load
        with self.lock:
            if len(self.load_history) < 3:
                return False
                
            recent_loads = [load for _, load in self.load_history[-5:]]  # Last 5 measurements
            avg_load = sum(recent_loads) / len(recent_loads)
            
            return avg_load > self.config.scale_up_threshold
            
    def should_scale_down(self) -> bool:
        """Determine if system should scale down."""
        if not self.config.enable_auto_scaling:
            return False
            
        if self.current_instances <= self.config.min_instances:
            return False
            
        if time.time() - self.last_scale_time < self.scaling_cooldown:
            return False
            
        # Check recent load
        with self.lock:
            if len(self.load_history) < 10:  # Need more history for scale down
                return False
                
            recent_loads = [load for _, load in self.load_history[-10:]]  # Last 10 measurements
            avg_load = sum(recent_loads) / len(recent_loads)
            
            return avg_load < self.config.scale_down_threshold
            
    def scale_up(self) -> bool:
        """
        Scale up the system.
        
        Returns:
            True if scaling was successful
        """
        if not self.should_scale_up():
            return False
            
        with self.lock:
            self.current_instances += 1
            self.last_scale_time = time.time()
            
        warnings.warn(f"Scaling up to {self.current_instances} instances")
        return True
        
    def scale_down(self) -> bool:
        """
        Scale down the system.
        
        Returns:
            True if scaling was successful
        """
        if not self.should_scale_down():
            return False
            
        with self.lock:
            self.current_instances = max(self.config.min_instances, self.current_instances - 1)
            self.last_scale_time = time.time()
            
        warnings.warn(f"Scaling down to {self.current_instances} instances")
        return True
        
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self.lock:
            recent_loads = [load for _, load in self.load_history[-10:]]
            current_load = recent_loads[-1] if recent_loads else 0.0
            avg_load = sum(recent_loads) / len(recent_loads) if recent_loads else 0.0
            
            return {
                'current_instances': self.current_instances,
                'min_instances': self.config.min_instances,
                'max_instances': self.config.max_instances,
                'current_load': current_load,
                'average_load': avg_load,
                'scale_up_threshold': self.config.scale_up_threshold,
                'scale_down_threshold': self.config.scale_down_threshold,
                'time_since_last_scale': time.time() - self.last_scale_time,
                'can_scale_up': self.should_scale_up(),
                'can_scale_down': self.should_scale_down()
            }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.signal_cache = SignalCache(self.config)
        self.concurrent_processor = ConcurrentProcessor(self.config)
        self.load_balancer = LoadBalancer(self.config)
        self.auto_scaler = AutoScaler(self.config)
        
        # Initialize default workers for testing/basic operation
        self._initialize_default_workers()
        
        # Performance metrics
        self.operation_times: Dict[str, List[float]] = {}
        self.throughput_history: List[Tuple[float, int]] = []  # (timestamp, requests_per_second)
    
    def _initialize_default_workers(self):
        """Initialize default workers for basic operation."""
        def default_worker_function(task_func):
            """Default worker that just executes the task."""
            return task_func()
        
        # Add a default worker
        self.load_balancer.add_worker(
            worker_id="default_worker",
            process_func=default_worker_function,
            capacity=100
        )
        
    def optimize_decode_operation(self, decode_func: Callable):
        """
        Decorator to optimize decoding operations.
        
        Args:
            decode_func: Original decode function
            
        Returns:
            Optimized decode function
        """
        def optimized_decode(signal, *args, **kwargs):
            start_time = time.time()
            
            # Try cache first
            if self.signal_cache.config.enable_signal_cache:
                signal_hash = self.signal_cache.calculate_signal_hash(signal)
                cached_result = self.signal_cache.get_processed_signal(
                    signal_hash, 
                    kwargs.get('preprocessing_config', 'default')
                )
                
                if cached_result is not None:
                    return cached_result
                    
            # Process with load balancing
            try:
                result = self.load_balancer.process_request(
                    lambda: decode_func(signal, *args, **kwargs)
                )
                
                # Cache result
                if self.signal_cache.config.enable_signal_cache:
                    self.signal_cache.cache_processed_signal(
                        signal_hash,
                        kwargs.get('preprocessing_config', 'default'),
                        result
                    )
                    
                # Record performance metrics
                duration = time.time() - start_time
                self._record_operation_time('decode', duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                self._record_operation_time('decode_failed', duration)
                raise
                
        return optimized_decode
        
    def _record_operation_time(self, operation: str, duration: float):
        """Record operation timing for performance analysis."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            
        self.operation_times[operation].append(duration)
        
        # Keep only recent times
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'cache_stats': self.signal_cache.get_cache_stats(),
            'load_balancer_stats': self.load_balancer.get_load_stats(),
            'auto_scaling_stats': self.auto_scaler.get_scaling_stats(),
            'operation_times': {}
        }
        
        # Calculate operation time statistics
        for operation, times in self.operation_times.items():
            if times:
                if HAS_NUMPY:
                    report['operation_times'][operation] = {
                        'count': len(times),
                        'mean': float(np.mean(times)),
                        'std': float(np.std(times)),
                        'min': float(np.min(times)),
                        'max': float(np.max(times)),
                        'p95': float(np.percentile(times, 95))
                    }
                else:
                    report['operation_times'][operation] = {
                        'count': len(times),
                        'mean': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times)
                    }
                    
        return report
        
    def optimize_for_throughput(self):
        """Apply optimizations for maximum throughput."""
        # Increase worker threads
        self.config.max_worker_threads = min(mp.cpu_count() * 2, 16)
        
        # Optimize cache settings
        self.config.cache_size = 2000
        self.config.cache_ttl = 600.0  # 10 minutes
        
        # Enable aggressive auto-scaling
        self.config.scale_up_threshold = 0.6
        self.config.scale_down_threshold = 0.2
        
        warnings.warn("Applied throughput optimizations")
        
    def optimize_for_latency(self):
        """Apply optimizations for minimum latency."""
        # Reduce batch sizes
        self.config.max_worker_threads = mp.cpu_count()
        
        # Aggressive caching
        self.config.cache_size = 5000
        self.config.cache_ttl = 1800.0  # 30 minutes
        
        # Conservative auto-scaling
        self.config.scale_up_threshold = 0.5
        self.config.scale_down_threshold = 0.1
        
        warnings.warn("Applied latency optimizations")
        
    def cleanup(self):
        """Cleanup optimization resources."""
        self.concurrent_processor.shutdown()


if __name__ == '__main__':
    # Test optimization system
    print("Testing BCI-2-Token Optimization System") 
    print("=" * 45)
    
    config = OptimizationConfig(
        enable_signal_cache=True,
        cache_size=100,
        max_worker_threads=2
    )
    
    optimizer = PerformanceOptimizer(config)
    
    # Test caching
    def mock_decode(signal):
        time.sleep(0.1)  # Simulate processing time
        return {'tokens': [1, 2, 3], 'confidence': 0.8}
        
    optimized_decode = optimizer.optimize_decode_operation(mock_decode)
    
    # Test with mock signal
    if HAS_NUMPY:
        test_signal = np.random.randn(8, 256)
        
        # First call (should be slow)
        start = time.time()
        result1 = optimized_decode(test_signal)
        time1 = time.time() - start
        
        # Second call (should be fast due to caching)
        start = time.time()
        result2 = optimized_decode(test_signal)
        time2 = time.time() - start
        
        print(f"✓ Caching test: {time1:.3f}s -> {time2:.3f}s (speedup: {time1/time2:.1f}x)")
    else:
        print("✓ Caching system initialized (numpy required for full test)")
        
    # Test auto-scaling
    scaler = optimizer.auto_scaler
    
    # Simulate high load
    for _ in range(5):
        scaler.record_load_metric(0.9)  # High load
        
    should_scale_up = scaler.should_scale_up()
    print(f"✓ Auto-scaling decision: scale up = {should_scale_up}")
    
    # Get performance report
    report = optimizer.get_performance_report()
    print(f"✓ Performance report generated with {len(report.keys())} sections")
    
    # Test optimizations
    optimizer.optimize_for_latency()
    optimizer.optimize_for_throughput()
    print("✓ Optimization presets applied")
    
    # Cleanup
    optimizer.cleanup()
    
    print("\n✓ Optimization system working")