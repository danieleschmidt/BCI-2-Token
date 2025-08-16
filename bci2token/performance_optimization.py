"""
Advanced Performance Optimization and Scaling for BCI-2-Token

This module implements sophisticated performance optimization techniques including:
- Intelligent caching with TTL and LRU policies
- Concurrent processing with thread/process pools
- Memory optimization and garbage collection
- Auto-scaling based on load metrics
- Performance profiling and bottleneck detection
"""

import time
import threading
import multiprocessing as mp
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
from functools import wraps, lru_cache
import weakref
import gc
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    # Caching
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 300
    enable_memory_cache: bool = True
    enable_disk_cache: bool = False
    
    # Concurrency
    max_workers: int = None  # None = auto-detect
    use_processes: bool = False  # True for CPU-bound, False for I/O-bound
    batch_processing: bool = True
    optimal_batch_size: int = 32
    
    # Memory optimization
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.8  # 80% memory usage
    enable_gc_tuning: bool = True
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # CPU/memory threshold to scale up
    scale_down_threshold: float = 0.3  # CPU/memory threshold to scale down
    min_workers: int = 1
    max_workers_limit: int = 16


class CacheEntry:
    """Cache entry with TTL and usage tracking"""
    
    def __init__(self, key: str, value: Any, ttl: float):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 1
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the cached value and update statistics"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache entry statistics"""
        return {
            'key': self.key,
            'created_at': self.created_at,
            'ttl': self.ttl,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'age_seconds': time.time() - self.created_at
        }


class IntelligentCache:
    """Multi-level intelligent caching system with LRU and TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if not entry.is_expired():
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return entry.access(), True
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            self.misses += 1
            return None, False
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache"""
        ttl = ttl or self.default_ttl
        
        with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                del self.cache[key]
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.evictions += 1
            
            # Add new entry
            entry = CacheEntry(key, value, ttl)
            self.cache[key] = entry
    
    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'memory_usage_mb': self._estimate_memory_usage(),
                'oldest_entry_age': self._get_oldest_entry_age()
            }
    
    def _cleanup_expired(self) -> None:
        """Background thread to clean up expired entries"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                with self.lock:
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del self.cache[key]
                        self.evictions += 1
                
            except Exception:
                # Continue on any errors
                continue
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of cache in MB"""
        # Rough estimation - in practice, use memory_profiler or similar
        try:
            total_size = sum(
                len(str(entry.key)) + len(str(entry.value))
                for entry in self.cache.values()
            )
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_oldest_entry_age(self) -> float:
        """Get age of oldest entry in seconds"""
        if not self.cache:
            return 0.0
        
        oldest_entry = next(iter(self.cache.values()))
        return time.time() - oldest_entry.created_at


def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    cache = IntelligentCache(default_ttl=ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(kwargs.items())))}"
            
            # Try to get from cache
            cached_result, cache_hit = cache.get(cache_key)
            if cache_hit:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        # Attach cache management functions
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_statistics
        wrapper.cache_invalidate = cache.invalidate
        
        return wrapper
    
    return decorator


class ConcurrentProcessor:
    """Concurrent processing for BCI operations"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Determine optimal worker count
        if config.max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)
        else:
            self.max_workers = config.max_workers
        
        # Thread pools for different types of operations
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.cpu_executor = ProcessPoolExecutor(max_workers=self.max_workers) if config.use_processes else self.io_executor
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.active_operations = 0
    
    def process_batch(self, operation: Callable, data_batch: List[Any], 
                     operation_type: str = 'cpu') -> List[Any]:
        """Process batch of data concurrently"""
        
        if not data_batch:
            return []
        
        start_time = time.time()
        self.active_operations += len(data_batch)
        
        try:
            # Choose appropriate executor
            executor = self.cpu_executor if operation_type == 'cpu' else self.io_executor
            
            # Submit all tasks
            future_to_data = {
                executor.submit(operation, data_item): data_item
                for data_item in data_batch
            }
            
            # Collect results
            results = []
            for future in as_completed(future_to_data):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    results.append(None)  # Placeholder for failed operations
            
            # Record performance
            total_time = time.time() - start_time
            self.operation_times[f'{operation_type}_batch'].append(total_time)
            
            self.logger.debug(f"Processed batch of {len(data_batch)} items in {total_time:.3f}s")
            
            return results
            
        finally:
            self.active_operations -= len(data_batch)
    
    def process_streaming(self, operation: Callable, data_stream: queue.Queue,
                         result_queue: queue.Queue, operation_type: str = 'cpu'):
        """Process streaming data concurrently"""
        
        executor = self.cpu_executor if operation_type == 'cpu' else self.io_executor
        active_futures = {}
        
        try:
            while True:
                try:
                    # Get data from stream (with timeout)
                    data_item = data_stream.get(timeout=1.0)
                    
                    if data_item is None:  # Sentinel value to stop processing
                        break
                    
                    # Submit processing task
                    future = executor.submit(operation, data_item)
                    active_futures[future] = data_item
                    
                    # Check for completed futures
                    completed_futures = [f for f in active_futures if f.done()]
                    
                    for future in completed_futures:
                        try:
                            result = future.result()
                            result_queue.put(result)
                        except Exception as e:
                            self.logger.error(f"Streaming processing error: {e}")
                            result_queue.put(None)
                        
                        del active_futures[future]
                    
                except queue.Empty:
                    # Check for any remaining completed futures
                    completed_futures = [f for f in active_futures if f.done()]
                    
                    for future in completed_futures:
                        try:
                            result = future.result()
                            result_queue.put(result)
                        except Exception as e:
                            self.logger.error(f"Streaming processing error: {e}")
                            result_queue.put(None)
                        
                        del active_futures[future]
                    
                    # Continue to next iteration
                    continue
            
            # Wait for all remaining futures
            for future in active_futures:
                try:
                    result = future.result(timeout=10)
                    result_queue.put(result)
                except Exception as e:
                    self.logger.error(f"Final streaming processing error: {e}")
                    result_queue.put(None)
        
        except Exception as e:
            self.logger.error(f"Streaming processor error: {e}")
        
        finally:
            # Signal end of processing
            result_queue.put(None)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get concurrent processing performance statistics"""
        stats = {
            'active_operations': self.active_operations,
            'max_workers': self.max_workers,
            'operation_times': {}
        }
        
        # Calculate statistics for each operation type
        for op_type, times in self.operation_times.items():
            if times:
                stats['operation_times'][op_type] = {
                    'count': len(times),
                    'mean_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times)
                }
        
        return stats
    
    def shutdown(self):
        """Shutdown all executors"""
        self.io_executor.shutdown(wait=True)
        if self.cpu_executor != self.io_executor:
            self.cpu_executor.shutdown(wait=True)


class MemoryOptimizer:
    """Memory optimization and monitoring"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Memory monitoring
        self.memory_samples = deque(maxlen=100)
        self.gc_stats = {
            'collections': 0,
            'freed_objects': 0,
            'collection_times': []
        }
        
        # Configure garbage collection if enabled
        if config.enable_gc_tuning:
            self._tune_garbage_collection()
    
    def monitor_memory(self) -> Dict[str, Any]:
        """Monitor current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_stats = {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
            
            self.memory_samples.append(memory_stats)
            
            return memory_stats
            
        except ImportError:
            # Fallback without psutil
            return {
                'rss_mb': 100.0,  # Simulated
                'vms_mb': 200.0,
                'percent': 15.0,
                'available_mb': 8000.0
            }
    
    def optimize_memory(self, force_gc: bool = False) -> Dict[str, Any]:
        """Perform memory optimization"""
        start_time = time.time()
        
        # Get memory before optimization
        memory_before = self.monitor_memory()
        
        # Force garbage collection if requested or threshold exceeded
        if force_gc or memory_before['percent'] > self.config.memory_cleanup_threshold * 100:
            
            # Manual garbage collection
            collected_objects = 0
            for generation in range(3):  # Python has 3 GC generations
                collected = gc.collect(generation)
                collected_objects += collected
            
            # Record GC statistics
            gc_time = time.time() - start_time
            self.gc_stats['collections'] += 1
            self.gc_stats['freed_objects'] += collected_objects
            self.gc_stats['collection_times'].append(gc_time)
            
            self.logger.debug(f"GC freed {collected_objects} objects in {gc_time:.3f}s")
        
        # Get memory after optimization
        memory_after = self.monitor_memory()
        
        return {
            'memory_before_mb': memory_before['rss_mb'],
            'memory_after_mb': memory_after['rss_mb'],
            'freed_mb': memory_before['rss_mb'] - memory_after['rss_mb'],
            'gc_time': time.time() - start_time,
            'objects_freed': collected_objects if 'collected_objects' in locals() else 0
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        current_memory = self.monitor_memory()
        
        stats = {
            'current': current_memory,
            'gc_stats': self.gc_stats.copy(),
        }
        
        if self.memory_samples:
            recent_memory = list(self.memory_samples)[-10:]  # Last 10 samples
            stats['recent_memory'] = {
                'mean_rss_mb': np.mean([m['rss_mb'] for m in recent_memory]),
                'max_rss_mb': np.max([m['rss_mb'] for m in recent_memory]),
                'min_rss_mb': np.min([m['rss_mb'] for m in recent_memory]),
                'trend': self._calculate_memory_trend(recent_memory)
            }
        
        return stats
    
    def _tune_garbage_collection(self):
        """Tune garbage collection for better performance"""
        try:
            # Adjust GC thresholds for better performance
            # Default is (700, 10, 10) - we make it less aggressive
            gc.set_threshold(1000, 15, 15)
            
            # Enable automatic garbage collection
            gc.enable()
            
            self.logger.debug("Garbage collection tuned for performance")
            
        except Exception as e:
            self.logger.warning(f"Could not tune garbage collection: {e}")
    
    def _calculate_memory_trend(self, memory_samples: List[Dict[str, Any]]) -> str:
        """Calculate memory usage trend"""
        if len(memory_samples) < 3:
            return "insufficient_data"
        
        rss_values = [m['rss_mb'] for m in memory_samples]
        
        # Simple linear trend calculation
        x = np.arange(len(rss_values))
        slope = np.polyfit(x, rss_values, 1)[0]
        
        if slope > 1.0:
            return "increasing"
        elif slope < -1.0:
            return "decreasing"
        else:
            return "stable"


class AutoScaler:
    """Automatic scaling based on performance metrics"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.current_workers = config.min_workers
        self.scaling_history = deque(maxlen=50)
        
        # Performance monitoring
        self.cpu_samples = deque(maxlen=20)
        self.memory_samples = deque(maxlen=20)
        self.request_rates = deque(maxlen=20)
        
        # Scaling decisions
        self.last_scale_time = 0
        self.scale_cooldown = 60  # Minimum seconds between scaling decisions
    
    def should_scale_up(self, cpu_usage: float, memory_usage: float, 
                       request_rate: float) -> bool:
        """Determine if scaling up is needed"""
        
        # Record current metrics
        self.cpu_samples.append(cpu_usage)
        self.memory_samples.append(memory_usage)
        self.request_rates.append(request_rate)
        
        # Check cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check if already at maximum
        if self.current_workers >= self.config.max_workers_limit:
            return False
        
        # Scale up conditions
        avg_cpu = np.mean(list(self.cpu_samples)[-5:]) if self.cpu_samples else 0
        avg_memory = np.mean(list(self.memory_samples)[-5:]) if self.memory_samples else 0
        
        cpu_threshold_exceeded = avg_cpu > self.config.scale_up_threshold
        memory_threshold_exceeded = avg_memory > self.config.scale_up_threshold
        
        # High request rate also triggers scaling
        if self.request_rates:
            recent_rate = np.mean(list(self.request_rates)[-3:])
            rate_increasing = len(self.request_rates) > 1 and recent_rate > self.request_rates[-2]
        else:
            rate_increasing = False
        
        return cpu_threshold_exceeded or memory_threshold_exceeded or rate_increasing
    
    def should_scale_down(self, cpu_usage: float, memory_usage: float,
                         request_rate: float) -> bool:
        """Determine if scaling down is appropriate"""
        
        # Check cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check if already at minimum
        if self.current_workers <= self.config.min_workers:
            return False
        
        # Scale down conditions (more conservative)
        if len(self.cpu_samples) < 10 or len(self.memory_samples) < 10:
            return False  # Need sufficient data
        
        avg_cpu = np.mean(list(self.cpu_samples)[-10:])
        avg_memory = np.mean(list(self.memory_samples)[-10:])
        
        cpu_threshold_ok = avg_cpu < self.config.scale_down_threshold
        memory_threshold_ok = avg_memory < self.config.scale_down_threshold
        
        # Also check that request rate is stable/decreasing
        if len(self.request_rates) >= 5:
            recent_rates = list(self.request_rates)[-5:]
            rate_trend = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]
            rate_stable = rate_trend <= 0  # Decreasing or stable
        else:
            rate_stable = False
        
        return cpu_threshold_ok and memory_threshold_ok and rate_stable
    
    def scale_up(self) -> int:
        """Scale up the number of workers"""
        old_workers = self.current_workers
        
        # Increase workers (but not beyond limit)
        scale_factor = 1.5  # Increase by 50%
        new_workers = min(
            int(self.current_workers * scale_factor),
            self.config.max_workers_limit
        )
        
        if new_workers > self.current_workers:
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_up',
                'from_workers': old_workers,
                'to_workers': new_workers,
                'reason': 'high_load'
            })
            
            self.logger.info(f"Scaled up from {old_workers} to {new_workers} workers")
        
        return self.current_workers
    
    def scale_down(self) -> int:
        """Scale down the number of workers"""
        old_workers = self.current_workers
        
        # Decrease workers (but not below minimum)
        scale_factor = 0.8  # Decrease by 20%
        new_workers = max(
            int(self.current_workers * scale_factor),
            self.config.min_workers
        )
        
        if new_workers < self.current_workers:
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_down',
                'from_workers': old_workers,
                'to_workers': new_workers,
                'reason': 'low_load'
            })
            
            self.logger.info(f"Scaled down from {old_workers} to {new_workers} workers")
        
        return self.current_workers
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers_limit,
            'scaling_events': len(self.scaling_history),
            'recent_scaling': list(self.scaling_history)[-5:] if self.scaling_history else [],
            'last_scale_time': self.last_scale_time,
            'seconds_since_last_scale': time.time() - self.last_scale_time,
            'current_metrics': {
                'avg_cpu': np.mean(list(self.cpu_samples)) if self.cpu_samples else 0,
                'avg_memory': np.mean(list(self.memory_samples)) if self.memory_samples else 0,
                'avg_request_rate': np.mean(list(self.request_rates)) if self.request_rates else 0
            }
        }


class PerformanceProfiler:
    """Performance profiling and bottleneck detection"""
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.bottlenecks = []
        self.profiling_active = False
        
    def profile_operation(self, operation_name: str):
        """Decorator to profile function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.profiling_active:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                start_cpu = time.process_time()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    end_cpu = time.process_time()
                    
                    wall_time = end_time - start_time
                    cpu_time = end_cpu - start_cpu
                    
                    self.operation_times[operation_name].append({
                        'wall_time': wall_time,
                        'cpu_time': cpu_time,
                        'timestamp': time.time()
                    })
                    
                    self.operation_counts[operation_name] += 1
                    
                    # Check for potential bottlenecks
                    if wall_time > 1.0:  # Operations taking > 1 second
                        self.bottlenecks.append({
                            'operation': operation_name,
                            'wall_time': wall_time,
                            'cpu_time': cpu_time,
                            'timestamp': time.time()
                        })
            
            return wrapper
        return decorator
    
    def start_profiling(self):
        """Start performance profiling"""
        self.profiling_active = True
        self.operation_times.clear()
        self.operation_counts.clear()
        self.bottlenecks.clear()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results"""
        self.profiling_active = False
        
        # Analyze collected data
        results = {
            'total_operations': sum(self.operation_counts.values()),
            'operation_stats': {},
            'bottlenecks': self.bottlenecks.copy(),
            'recommendations': []
        }
        
        # Calculate statistics for each operation
        for operation, times in self.operation_times.items():
            if times:
                wall_times = [t['wall_time'] for t in times]
                cpu_times = [t['cpu_time'] for t in times]
                
                results['operation_stats'][operation] = {
                    'count': len(times),
                    'total_wall_time': sum(wall_times),
                    'mean_wall_time': np.mean(wall_times),
                    'max_wall_time': max(wall_times),
                    'total_cpu_time': sum(cpu_times),
                    'mean_cpu_time': np.mean(cpu_times),
                    'cpu_efficiency': np.mean([c/w for c, w in zip(cpu_times, wall_times) if w > 0])
                }
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['operation_stats'])
        
        return results
    
    def _generate_recommendations(self, operation_stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Find operations with high total time
        if operation_stats:
            sorted_ops = sorted(
                operation_stats.items(),
                key=lambda x: x[1]['total_wall_time'],
                reverse=True
            )
            
            # Top time consumer
            if sorted_ops:
                top_operation = sorted_ops[0]
                recommendations.append(
                    f"'{top_operation[0]}' consumes most time "
                    f"({top_operation[1]['total_wall_time']:.2f}s total)"
                )
            
            # Operations with low CPU efficiency (I/O bound)
            io_bound_ops = [
                (op, stats) for op, stats in operation_stats.items()
                if stats['cpu_efficiency'] < 0.5 and stats['mean_wall_time'] > 0.1
            ]
            
            if io_bound_ops:
                recommendations.append(
                    f"I/O bound operations detected: {[op for op, _ in io_bound_ops]}"
                )
                recommendations.append("Consider using async I/O or concurrent processing")
            
            # Operations with high variance (inconsistent performance)
            for op, stats in operation_stats.items():
                if stats['count'] > 5:  # Need sufficient samples
                    times = [t['wall_time'] for t in self.operation_times[op]]
                    cv = np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
                    
                    if cv > 0.5:  # High coefficient of variation
                        recommendations.append(
                            f"'{op}' has inconsistent performance (CV={cv:.2f})"
                        )
        
        # Bottleneck-based recommendations
        if self.bottlenecks:
            recommendations.append(f"{len(self.bottlenecks)} operations took >1s to complete")
            
            # Group bottlenecks by operation
            bottleneck_counts = defaultdict(int)
            for bottleneck in self.bottlenecks:
                bottleneck_counts[bottleneck['operation']] += 1
            
            if bottleneck_counts:
                worst_op = max(bottleneck_counts.items(), key=lambda x: x[1])
                recommendations.append(f"'{worst_op[0]}' was slow {worst_op[1]} times")
        
        return recommendations


# Example usage and demonstration
def demo_performance_optimization():
    """Demonstrate performance optimization capabilities"""
    print("=== Performance Optimization Demo ===\n")
    
    config = PerformanceConfig(
        cache_max_size=100,
        max_workers=4,
        enable_memory_monitoring=True,
        enable_auto_scaling=True
    )
    
    # 1. Caching Demo
    print("1. Intelligent Caching")
    
    @cached(ttl=60)
    def expensive_computation(x: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x ** 2
    
    # Test caching
    start_time = time.time()
    result1 = expensive_computation(5)  # Cache miss
    time1 = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_computation(5)  # Cache hit
    time2 = time.time() - start_time
    
    print(f"  First call: {time1:.3f}s, Second call: {time2:.3f}s")
    print(f"  Cache stats: {expensive_computation.cache_stats()}")
    
    # 2. Concurrent Processing Demo
    print("\n2. Concurrent Processing")
    
    processor = ConcurrentProcessor(config)
    
    def simulate_processing(data):
        time.sleep(0.05)  # Simulate work
        return data * 2
    
    # Process batch
    test_data = list(range(20))
    start_time = time.time()
    results = processor.process_batch(simulate_processing, test_data, 'cpu')
    batch_time = time.time() - start_time
    
    print(f"  Processed {len(test_data)} items in {batch_time:.3f}s")
    print(f"  Performance: {processor.get_performance_stats()}")
    
    # 3. Memory Optimization Demo
    print("\n3. Memory Optimization")
    
    memory_optimizer = MemoryOptimizer(config)
    
    # Create some objects to demonstrate GC
    large_objects = []
    for i in range(1000):
        large_objects.append([np.random.randn(100) for _ in range(10)])
    
    memory_before = memory_optimizer.monitor_memory()
    
    # Clear objects and optimize memory
    large_objects.clear()
    optimization_result = memory_optimizer.optimize_memory(force_gc=True)
    
    print(f"  Memory before: {memory_before['rss_mb']:.1f} MB")
    print(f"  Memory after: {optimization_result['memory_after_mb']:.1f} MB")
    print(f"  Freed: {optimization_result['freed_mb']:.1f} MB")
    
    # 4. Performance Profiling Demo
    print("\n4. Performance Profiling")
    
    profiler = PerformanceProfiler()
    
    @profiler.profile_operation('fast_operation')
    def fast_operation(x):
        return x * 2
    
    @profiler.profile_operation('slow_operation')
    def slow_operation(x):
        time.sleep(0.01)
        return x ** 2
    
    profiler.start_profiling()
    
    # Run operations
    for i in range(50):
        fast_operation(i)
        if i % 10 == 0:
            slow_operation(i)
    
    profiling_results = profiler.stop_profiling()
    
    print(f"  Total operations: {profiling_results['total_operations']}")
    print(f"  Bottlenecks detected: {len(profiling_results['bottlenecks'])}")
    print(f"  Recommendations: {len(profiling_results['recommendations'])}")
    
    for rec in profiling_results['recommendations'][:3]:  # Show first 3
        print(f"    - {rec}")
    
    # Cleanup
    processor.shutdown()
    
    print("\n=== Performance Optimization Demo Complete ===")


class HyperscaleOptimizer:
    """Generation 3 Hyperscale Performance Optimizer with advanced ML-driven optimization."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.performance_history = deque(maxlen=10000)
        self.optimization_patterns = {}
        self.adaptive_thresholds = {}
        self.ml_predictions = {}
        self.lock = threading.Lock()
        
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns using machine learning techniques."""
        if len(self.performance_history) < 100:
            return {'status': 'insufficient_data', 'samples': len(self.performance_history)}
            
        with self.lock:
            # Convert to numpy arrays for analysis
            timestamps = [entry['timestamp'] for entry in self.performance_history]
            cpu_usage = [entry['cpu'] for entry in self.performance_history]
            memory_usage = [entry['memory'] for entry in self.performance_history]
            request_rates = [entry['request_rate'] for entry in self.performance_history]
            response_times = [entry['response_time'] for entry in self.performance_history]
            
            # Time-based analysis
            current_time = time.time()
            time_deltas = [(current_time - ts) / 3600 for ts in timestamps]  # Hours ago
            
            # Detect patterns
            patterns = {
                'peak_hours': self._detect_peak_hours(timestamps, request_rates),
                'cpu_memory_correlation': np.corrcoef(cpu_usage, memory_usage)[0, 1],
                'load_response_correlation': np.corrcoef(request_rates, response_times)[0, 1],
                'daily_patterns': self._detect_daily_patterns(timestamps, cpu_usage),
                'anomalies': self._detect_performance_anomalies(cpu_usage, memory_usage, response_times)
            }
            
            # Predictive modeling
            predictions = self._generate_predictions(timestamps, cpu_usage, memory_usage, request_rates)
            
            return {
                'status': 'analysis_complete',
                'samples_analyzed': len(self.performance_history),
                'patterns': patterns,
                'predictions': predictions,
                'recommendations': self._generate_optimization_recommendations(patterns, predictions)
            }
            
    def _detect_peak_hours(self, timestamps: List[float], request_rates: List[float]) -> Dict[str, Any]:
        """Detect peak usage hours for proactive scaling."""
        import datetime
        
        hourly_rates = defaultdict(list)
        for ts, rate in zip(timestamps, request_rates):
            hour = datetime.datetime.fromtimestamp(ts).hour
            hourly_rates[hour].append(rate)
            
        # Calculate average rates per hour
        avg_hourly_rates = {hour: np.mean(rates) for hour, rates in hourly_rates.items()}
        
        # Find peak hours (top 25%)
        sorted_hours = sorted(avg_hourly_rates.items(), key=lambda x: x[1], reverse=True)
        peak_threshold = np.percentile(list(avg_hourly_rates.values()), 75)
        peak_hours = [hour for hour, rate in sorted_hours if rate >= peak_threshold]
        
        return {
            'peak_hours': peak_hours,
            'hourly_averages': avg_hourly_rates,
            'peak_threshold': peak_threshold
        }
        
    def _detect_daily_patterns(self, timestamps: List[float], metrics: List[float]) -> Dict[str, Any]:
        """Detect daily usage patterns."""
        import datetime
        
        daily_data = defaultdict(list)
        for ts, metric in zip(timestamps, metrics):
            day_of_week = datetime.datetime.fromtimestamp(ts).weekday()  # 0=Monday, 6=Sunday
            daily_data[day_of_week].append(metric)
            
        daily_averages = {day: np.mean(values) for day, values in daily_data.items()}
        
        # Identify weekday vs weekend patterns
        weekday_avg = np.mean([avg for day, avg in daily_averages.items() if day < 5])
        weekend_avg = np.mean([avg for day, avg in daily_averages.items() if day >= 5])
        
        return {
            'daily_averages': daily_averages,
            'weekday_avg': weekday_avg,
            'weekend_avg': weekend_avg,
            'weekend_weekday_ratio': weekend_avg / weekday_avg if weekday_avg > 0 else 1.0
        }
        
    def _detect_performance_anomalies(self, cpu_usage: List[float], 
                                    memory_usage: List[float], 
                                    response_times: List[float]) -> Dict[str, Any]:
        """Detect performance anomalies using statistical methods."""
        anomalies = {
            'cpu_anomalies': [],
            'memory_anomalies': [],
            'response_time_anomalies': []
        }
        
        # Use z-score for anomaly detection (values > 3 standard deviations)
        for name, data in [('cpu', cpu_usage), ('memory', memory_usage), ('response_time', response_times)]:
            if len(data) < 10:
                continue
                
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val > 0:
                z_scores = [(val - mean_val) / std_val for val in data]
                anomaly_indices = [i for i, z in enumerate(z_scores) if abs(z) > 3]
                
                anomalies[f'{name}_anomalies'] = [
                    {'index': i, 'value': data[i], 'z_score': z_scores[i]}
                    for i in anomaly_indices
                ]
                
        return anomalies
        
    def _generate_predictions(self, timestamps: List[float], cpu_usage: List[float],
                            memory_usage: List[float], request_rates: List[float]) -> Dict[str, Any]:
        """Generate performance predictions using simple time series analysis."""
        predictions = {}
        
        # Simple linear trend prediction for next hour
        current_time = time.time()
        future_time = current_time + 3600  # 1 hour ahead
        
        for name, data in [('cpu', cpu_usage), ('memory', memory_usage), ('request_rate', request_rates)]:
            if len(data) >= 10:
                # Use last 10 points for trend
                recent_data = data[-10:]
                x_values = list(range(len(recent_data)))
                
                # Linear regression
                coeffs = np.polyfit(x_values, recent_data, 1)
                trend_slope = coeffs[0]
                current_value = recent_data[-1]
                
                # Predict next 6 values (next hour in 10-minute intervals)
                future_predictions = []
                for i in range(1, 7):
                    predicted_value = current_value + (trend_slope * i)
                    future_predictions.append(max(0, predicted_value))  # Ensure non-negative
                    
                predictions[name] = {
                    'current': current_value,
                    'trend_slope': trend_slope,
                    'predictions_1h': future_predictions,
                    'avg_predicted': np.mean(future_predictions)
                }
                
        return predictions
        
    def _generate_optimization_recommendations(self, patterns: Dict[str, Any], 
                                             predictions: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Peak hours recommendations
        if 'peak_hours' in patterns:
            peak_hours = patterns['peak_hours']['peak_hours']
            if peak_hours:
                recommendations.append(
                    f"Consider proactive scaling during peak hours: {peak_hours}"
                )
                
        # CPU/Memory correlation
        if 'cpu_memory_correlation' in patterns:
            correlation = patterns['cpu_memory_correlation']
            if correlation > 0.8:
                recommendations.append(
                    "High CPU-memory correlation detected. Consider memory optimization."
                )
            elif correlation < 0.2:
                recommendations.append(
                    "Low CPU-memory correlation. May have separate bottlenecks."
                )
                
        # Response time predictions
        if 'response_time' in predictions:
            predicted_avg = predictions['response_time']['avg_predicted']
            current = predictions['response_time']['current']
            
            if predicted_avg > current * 1.5:
                recommendations.append(
                    "Response times predicted to increase significantly. Consider scaling up."
                )
            elif predicted_avg < current * 0.7:
                recommendations.append(
                    "Response times predicted to improve. Consider scaling down to save resources."
                )
                
        # Anomaly-based recommendations
        if 'anomalies' in patterns:
            cpu_anomalies = len(patterns['anomalies']['cpu_anomalies'])
            memory_anomalies = len(patterns['anomalies']['memory_anomalies'])
            
            if cpu_anomalies > 5:
                recommendations.append(
                    f"High number of CPU anomalies ({cpu_anomalies}). Investigate CPU spikes."
                )
            if memory_anomalies > 5:
                recommendations.append(
                    f"High number of memory anomalies ({memory_anomalies}). Check for memory leaks."
                )
                
        return recommendations
        
    def record_performance_sample(self, cpu: float, memory: float, 
                                request_rate: float, response_time: float):
        """Record a performance sample for analysis."""
        with self.lock:
            self.performance_history.append({
                'timestamp': time.time(),
                'cpu': cpu,
                'memory': memory,
                'request_rate': request_rate,
                'response_time': response_time
            })
            
    def optimize_based_on_predictions(self) -> Dict[str, Any]:
        """Optimize system configuration based on ML predictions."""
        analysis = self.analyze_performance_patterns()
        
        if analysis['status'] != 'analysis_complete':
            return analysis
            
        optimizations = {
            'cache_adjustments': {},
            'scaling_adjustments': {},
            'threshold_adjustments': {}
        }
        
        predictions = analysis['predictions']
        
        # Optimize cache based on memory predictions
        if 'memory' in predictions:
            predicted_memory = predictions['memory']['avg_predicted']
            current_memory = predictions['memory']['current']
            
            if predicted_memory > 0.9:  # High memory usage predicted
                # Reduce cache size
                new_cache_size = max(100, int(self.config.cache_max_size * 0.8))
                optimizations['cache_adjustments']['max_size'] = new_cache_size
                
        # Optimize scaling thresholds based on patterns
        patterns = analysis['patterns']
        if 'peak_hours' in patterns:
            current_hour = time.localtime().tm_hour
            if current_hour in patterns['peak_hours']['peak_hours']:
                # Lower scale-up threshold during peak hours
                optimizations['scaling_adjustments']['scale_up_threshold'] = 0.6
            else:
                # Higher threshold during off-peak
                optimizations['scaling_adjustments']['scale_up_threshold'] = 0.8
                
        return {
            'status': 'optimizations_generated',
            'optimizations': optimizations,
            'analysis': analysis
        }


class AdaptiveLoadBalancer:
    """Generation 3 Adaptive Load Balancer with ML-driven routing decisions."""
    
    def __init__(self, workers: List[str]):
        self.workers = workers
        self.worker_stats = {worker: {
            'load': 0.0,
            'response_times': deque(maxlen=100),
            'error_rates': deque(maxlen=100),
            'last_used': 0.0
        } for worker in workers}
        self.routing_algorithm = 'adaptive_weighted'
        self.lock = threading.Lock()
        
    def select_worker(self, request_complexity: float = 1.0) -> str:
        """Select optimal worker using adaptive algorithm."""
        with self.lock:
            if self.routing_algorithm == 'adaptive_weighted':
                return self._adaptive_weighted_selection(request_complexity)
            elif self.routing_algorithm == 'least_response_time':
                return self._least_response_time_selection()
            else:  # Default to round-robin
                return self._round_robin_selection()
                
    def _adaptive_weighted_selection(self, request_complexity: float) -> str:
        """Select worker based on adaptive weighting algorithm."""
        scores = {}
        current_time = time.time()
        
        for worker in self.workers:
            stats = self.worker_stats[worker]
            
            # Calculate adaptive score
            load_score = 1.0 - stats['load']  # Lower load = higher score
            
            # Response time score
            if stats['response_times']:
                avg_response_time = np.mean(stats['response_times'])
                max_response_time = max(stats['response_times'])
                response_score = 1.0 / (1.0 + avg_response_time)
            else:
                response_score = 1.0
                
            # Error rate score
            if stats['error_rates']:
                avg_error_rate = np.mean(stats['error_rates'])
                error_score = 1.0 - avg_error_rate
            else:
                error_score = 1.0
                
            # Freshness score (prefer workers not used recently)
            time_since_last_use = current_time - stats['last_used']
            freshness_score = min(1.0, time_since_last_use / 60.0)  # Max benefit after 1 minute
            
            # Complexity adjustment
            complexity_factor = 1.0 + (request_complexity - 1.0) * 0.5
            
            # Weighted combination
            final_score = (
                load_score * 0.4 +
                response_score * 0.3 +
                error_score * 0.2 +
                freshness_score * 0.1
            ) * complexity_factor
            
            scores[worker] = final_score
            
        # Select worker with highest score
        best_worker = max(scores.items(), key=lambda x: x[1])[0]
        self.worker_stats[best_worker]['last_used'] = current_time
        
        return best_worker
        
    def _least_response_time_selection(self) -> str:
        """Select worker with lowest average response time."""
        best_worker = self.workers[0]
        best_time = float('inf')
        
        for worker in self.workers:
            stats = self.worker_stats[worker]
            if stats['response_times']:
                avg_time = np.mean(stats['response_times'])
            else:
                avg_time = 0.0
                
            if avg_time < best_time:
                best_time = avg_time
                best_worker = worker
                
        return best_worker
        
    def _round_robin_selection(self) -> str:
        """Simple round-robin selection."""
        # Find least recently used worker
        return min(self.workers, key=lambda w: self.worker_stats[w]['last_used'])
        
    def update_worker_stats(self, worker: str, response_time: float, 
                          error_occurred: bool, current_load: float):
        """Update worker statistics."""
        with self.lock:
            if worker in self.worker_stats:
                stats = self.worker_stats[worker]
                stats['response_times'].append(response_time)
                stats['error_rates'].append(1.0 if error_occurred else 0.0)
                stats['load'] = current_load
                
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            worker_summaries = {}
            for worker, stats in self.worker_stats.items():
                avg_response_time = np.mean(stats['response_times']) if stats['response_times'] else 0.0
                avg_error_rate = np.mean(stats['error_rates']) if stats['error_rates'] else 0.0
                
                worker_summaries[worker] = {
                    'current_load': stats['load'],
                    'avg_response_time': avg_response_time,
                    'avg_error_rate': avg_error_rate,
                    'requests_processed': len(stats['response_times']),
                    'last_used': stats['last_used']
                }
                
            return {
                'algorithm': self.routing_algorithm,
                'total_workers': len(self.workers),
                'worker_stats': worker_summaries
            }


if __name__ == "__main__":
    demo_performance_optimization()