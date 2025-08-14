"""
Performance scaling and optimization for BCI-2-Token.

Provides caching, concurrent processing, load balancing, and performance
optimization features for high-scale deployments.
"""

import time
import threading
import multiprocessing as mp
import queue
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import functools
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging

import numpy as np
from .utils import BCIError


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    RANDOM = "random"     # Random replacement


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class IntelligentCache:
    """
    High-performance cache with multiple strategies and automatic optimization.
    """
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU,
                 ttl: float = 3600.0, auto_optimize: bool = True):
        self.max_size = max_size
        self.strategy = strategy
        self.ttl = ttl
        self.auto_optimize = auto_optimize
        
        # Cache storage
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Auto-optimization
        self.optimization_interval = 100  # Optimize every N operations
        self.operation_count = 0
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if self.strategy != CacheStrategy.TTL:
            return False
        
        entry = self.cache.get(key)
        if not entry:
            return True
            
        return time.time() - entry['timestamp'] > self.ttl
    
    def _evict_entry(self) -> Optional[str]:
        """Evict an entry based on strategy."""
        if not self.cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
            if expired_keys:
                oldest_key = expired_keys[0]
            else:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                
        else:  # RANDOM
            import random
            oldest_key = random.choice(list(self.cache.keys()))
        
        # Remove entry
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        del self.access_counts[oldest_key]
        self.stats.evictions += 1
        
        return oldest_key
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self.operation_count += 1
            
            # Check if key exists and not expired
            if key in self.cache and not self._is_expired(key):
                # Update access statistics
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                self.stats.hits += 1
                return self.cache[key]['value']
            else:
                # Remove expired entry
                if key in self.cache:
                    del self.cache[key]
                    del self.access_times[key]
                    del self.access_counts[key]
                
                self.stats.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache."""
        with self.lock:
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                evicted_key = self._evict_entry()
                if not evicted_key:
                    break
            
            # Store value
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'size': self._estimate_size(value)
            }
            
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            
            self.stats.total_size = len(self.cache)
            
            # Auto-optimize if enabled
            if (self.auto_optimize and 
                self.operation_count % self.optimization_interval == 0):
                self._optimize_strategy()
            
            return True
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 100  # Default estimate
    
    def _optimize_strategy(self):
        """Automatically optimize cache strategy based on performance."""
        if self.stats.hits + self.stats.misses < 50:
            return  # Not enough data
        
        current_hit_rate = self.stats.hit_rate
        
        # Try different strategies and pick the best
        strategies_to_try = [s for s in CacheStrategy if s != self.strategy]
        
        # This is a simplified optimization - in practice, you'd want
        # more sophisticated analysis
        if current_hit_rate < 0.5:  # Poor hit rate
            if self.strategy != CacheStrategy.LFU:
                logging.info(f"Cache optimization: switching to LFU (hit rate: {current_hit_rate:.2%})")
                self.strategy = CacheStrategy.LFU
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            self.stats.total_size = len(self.cache)
            return self.stats


class ProcessingPool:
    """
    High-performance processing pool with dynamic scaling.
    """
    
    def __init__(self, max_workers: int = None, use_processes: bool = False,
                 auto_scale: bool = True):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.auto_scale = auto_scale
        
        # Current pool
        self.executor = None
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Performance tracking
        self.task_times = []
        self.max_history = 100
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Auto-scaling parameters
        self.scale_up_threshold = 0.8    # Scale up if utilization > 80%
        self.scale_down_threshold = 0.3  # Scale down if utilization < 30%
        self.min_workers = 2
        
    def _create_executor(self, num_workers: int):
        """Create executor with specified number of workers."""
        if self.executor:
            self.executor.shutdown(wait=False)
        
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        self.max_workers = num_workers
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task for processing."""
        with self.lock:
            if not self.executor:
                self._create_executor(self.max_workers)
            
            self.active_tasks += 1
            
            # Wrap function to track performance
            def wrapped_func(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.completed_tasks += 1
                    return result
                except Exception as e:
                    self.failed_tasks += 1
                    raise e
                finally:
                    duration = time.time() - start_time
                    with self.lock:
                        self.active_tasks -= 1
                        self.task_times.append(duration)
                        if len(self.task_times) > self.max_history:
                            self.task_times.pop(0)
            
            future = self.executor.submit(wrapped_func, *args, **kwargs)
            
            # Auto-scale if enabled
            if self.auto_scale:
                self._check_scaling()
            
            return future
    
    def submit_batch(self, func: Callable, batch_data: List[Any]) -> List[Any]:
        """Submit a batch of tasks and wait for all to complete."""
        futures = []
        
        for data in batch_data:
            if isinstance(data, (list, tuple)):
                future = self.submit_task(func, *data)
            else:
                future = self.submit_task(func, data)
            futures.append(future)
        
        # Wait for all tasks to complete
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    def _check_scaling(self):
        """Check if pool should be scaled."""
        if not self.task_times:
            return
        
        # Calculate current utilization
        utilization = self.active_tasks / self.max_workers
        avg_task_time = np.mean(self.task_times[-10:])  # Last 10 tasks
        
        # Scale up if high utilization and slow tasks
        if (utilization > self.scale_up_threshold and 
            avg_task_time > 0.1 and  # Tasks taking > 100ms
            self.max_workers < mp.cpu_count() * 2):
            
            new_workers = min(self.max_workers + 2, mp.cpu_count() * 2)
            logging.info(f"Scaling up processing pool: {self.max_workers} -> {new_workers}")
            self._create_executor(new_workers)
            
        # Scale down if low utilization
        elif (utilization < self.scale_down_threshold and 
              self.max_workers > self.min_workers):
            
            new_workers = max(self.max_workers - 1, self.min_workers)
            logging.info(f"Scaling down processing pool: {self.max_workers} -> {new_workers}")
            self._create_executor(new_workers)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing pool statistics."""
        with self.lock:
            avg_task_time = np.mean(self.task_times) if self.task_times else 0
            utilization = self.active_tasks / self.max_workers
            
            return {
                'max_workers': self.max_workers,
                'active_tasks': self.active_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'utilization': utilization,
                'avg_task_time': avg_task_time,
                'success_rate': self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks)
            }
    
    def shutdown(self):
        """Shutdown the processing pool."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None


class LoadBalancer:
    """
    Simple load balancer for distributing work across multiple processors.
    """
    
    def __init__(self, processors: List[Callable]):
        self.processors = processors
        self.processor_stats = {i: {'requests': 0, 'total_time': 0.0, 'errors': 0} 
                               for i in range(len(processors))}
        self.current_index = 0
        self.lock = threading.Lock()
    
    def get_next_processor(self) -> Tuple[int, Callable]:
        """Get next processor using round-robin with load consideration."""
        with self.lock:
            # Find processor with lowest load
            best_processor = min(self.processor_stats.keys(), 
                               key=lambda i: self._get_load_score(i))
            
            processor = self.processors[best_processor]
            return best_processor, processor
    
    def _get_load_score(self, processor_id: int) -> float:
        """Calculate load score for processor."""
        stats = self.processor_stats[processor_id]
        if stats['requests'] == 0:
            return 0.0
        
        avg_time = stats['total_time'] / stats['requests']
        error_rate = stats['errors'] / stats['requests']
        
        # Combine response time and error rate
        return avg_time * (1 + error_rate * 10)
    
    def process_with_balancing(self, *args, **kwargs) -> Any:
        """Process request with load balancing."""
        processor_id, processor = self.get_next_processor()
        
        start_time = time.time()
        try:
            result = processor(*args, **kwargs)
            
            # Update statistics
            with self.lock:
                stats = self.processor_stats[processor_id]
                stats['requests'] += 1
                stats['total_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            # Update error statistics
            with self.lock:
                stats = self.processor_stats[processor_id]
                stats['requests'] += 1
                stats['total_time'] += time.time() - start_time
                stats['errors'] += 1
            
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            total_requests = sum(stats['requests'] for stats in self.processor_stats.values())
            
            return {
                'total_requests': total_requests,
                'processor_stats': dict(self.processor_stats),
                'processors_count': len(self.processors)
            }


class PerformanceOptimizer:
    """
    Comprehensive performance optimization coordinator.
    """
    
    def __init__(self):
        self.cache = IntelligentCache(max_size=1000, auto_optimize=True)
        self.processing_pool = ProcessingPool(auto_scale=True)
        self.load_balancer = None
        
        # Memory management
        self.gc_threshold = 1024 * 1024 * 100  # 100MB
        self.last_gc_time = time.time()
        self.gc_interval = 60.0  # 1 minute
        
    def cached_operation(self, cache_key: str = None, ttl: float = None):
        """Decorator for caching expensive operations."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key:
                    key = cache_key
                else:
                    key = f"{func.__name__}_{self.cache._generate_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_result = self.cache.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache.put(key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def parallel_operation(self, use_processes: bool = False):
        """Decorator for parallel processing."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(data_list, *args, **kwargs):
                if len(data_list) == 1:
                    # Single item, no need for parallel processing
                    return [func(data_list[0], *args, **kwargs)]
                
                # Create processing function
                def process_item(item):
                    return func(item, *args, **kwargs)
                
                # Use parallel processing
                return self.processing_pool.submit_batch(process_item, data_list)
            
            return wrapper
        return decorator
    
    def memory_managed_operation(self, cleanup_threshold: int = None):
        """Decorator for memory management."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Check if garbage collection is needed
                current_time = time.time()
                if (current_time - self.last_gc_time > self.gc_interval):
                    self._perform_memory_cleanup()
                    self.last_gc_time = current_time
                
                result = func(*args, **kwargs)
                
                return result
            
            return wrapper
        return decorator
    
    def _perform_memory_cleanup(self):
        """Perform memory cleanup."""
        # Force garbage collection
        collected = gc.collect()
        
        # Clear old cache entries if cache is large
        cache_stats = self.cache.get_stats()
        if cache_stats.total_size > 500:  # If cache has > 500 items
            # Clear 25% of cache
            keys_to_remove = list(self.cache.cache.keys())[:len(self.cache.cache) // 4]
            for key in keys_to_remove:
                if key in self.cache.cache:
                    del self.cache.cache[key]
                    del self.cache.access_times[key]
                    del self.cache.access_counts[key]
        
        logging.debug(f"Memory cleanup: collected {collected} objects")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_stats = self.cache.get_stats()
        pool_stats = self.processing_pool.get_stats()
        
        report = {
            'cache_stats': {
                'hit_rate': cache_stats.hit_rate,
                'total_size': cache_stats.total_size,
                'hits': cache_stats.hits,
                'misses': cache_stats.misses,
                'evictions': cache_stats.evictions
            },
            'processing_pool': pool_stats,
            'memory_info': {
                'gc_enabled': gc.isenabled(),
                'gc_counts': gc.get_count(),
            }
        }
        
        if self.load_balancer:
            report['load_balancer'] = self.load_balancer.get_stats()
        
        return report
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize performance for specific workload."""
        if workload_type == "high_throughput":
            # Optimize for high throughput
            self.cache.max_size = 2000
            self.cache.strategy = CacheStrategy.LFU
            self.processing_pool.max_workers = mp.cpu_count() * 2
            
        elif workload_type == "low_latency":
            # Optimize for low latency
            self.cache.max_size = 500
            self.cache.strategy = CacheStrategy.LRU
            self.processing_pool.max_workers = mp.cpu_count()
            
        elif workload_type == "memory_constrained":
            # Optimize for memory efficiency
            self.cache.max_size = 200
            self.cache.strategy = CacheStrategy.TTL
            self.cache.ttl = 300.0  # 5 minutes
            self.processing_pool.max_workers = max(2, mp.cpu_count() // 2)
        
        logging.info(f"Performance optimized for {workload_type} workload")


# Global performance optimizer instance
_performance_optimizer = PerformanceOptimizer()


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    return _performance_optimizer


def cached(cache_key: str = None, ttl: float = None):
    """Convenience decorator for caching."""
    return _performance_optimizer.cached_operation(cache_key, ttl)


def parallel(use_processes: bool = False):
    """Convenience decorator for parallel processing."""
    return _performance_optimizer.parallel_operation(use_processes)