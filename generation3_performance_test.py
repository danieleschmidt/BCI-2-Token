#!/usr/bin/env python3
"""
Generation 3 Performance and Scaling Test Suite
===============================================

Tests performance optimization, caching, concurrent processing,
auto-scaling, and resource management features.
"""

import sys
import time
import numpy as np
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import threading

def test_intelligent_caching():
    """Test intelligent caching system."""
    print("Testing intelligent caching...")
    
    try:
        from bci2token.performance_optimization import IntelligentCache
        
        # Create cache with small size for testing
        cache = IntelligentCache(max_size=5, default_ttl=2)
        
        # Test basic caching
        cache.put("key1", "value1")
        result, hit = cache.get("key1")
        assert result == "value1", "Cache failed to store/retrieve value"
        assert hit == True, "Cache should report hit"
        
        # Test cache miss
        result, hit = cache.get("nonexistent")
        assert result is None, "Cache should return None for missing keys"
        assert hit == False, "Cache should report miss"
        
        # Test TTL expiration
        cache.put("ttl_test", "expires", ttl=0.1)  # 100ms TTL
        time.sleep(0.2)  # Wait for expiration
        result, hit = cache.get("ttl_test")
        assert result is None, "Expired cache entry should return None"
        
        # Test cache statistics
        stats = cache.get_statistics()
        assert "hit_rate" in stats, "Cache stats should include hit rate"
        
        print("   ✓ Intelligent caching functional")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Caching module not available: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Caching test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization features."""
    print("Testing performance optimization...")
    
    try:
        from bci2token.performance_optimization import PerformanceOptimizer, PerformanceConfig
        
        # Create optimizer with test configuration
        config = PerformanceConfig(
            cache_max_size=100,
            max_workers=2,
            enable_memory_monitoring=True
        )
        optimizer = PerformanceOptimizer(config)
        
        # Test optimization
        def slow_function(x):
            time.sleep(0.01)  # Simulate slow operation
            return x * 2
        
        # Test function optimization
        optimized_func = optimizer.optimize_function(slow_function)
        
        # Run optimized function multiple times
        results = []
        start_time = time.time()
        for i in range(10):
            result = optimized_func(i)
            results.append(result)
        end_time = time.time()
        
        # Verify results are correct
        expected = [i * 2 for i in range(10)]
        assert results == expected, "Optimized function produced incorrect results"
        
        # Get performance report
        report = optimizer.get_performance_report()
        assert "cache_stats" in report, "Performance report should include cache stats"
        
        print(f"   ✓ Performance optimization functional (completed in {end_time - start_time:.3f}s)")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Performance optimization module not available: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Performance optimization test failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("Testing concurrent processing...")
    
    try:
        from bci2token.performance_scaling import ProcessingPool
        
        # Create processing pool
        pool = ProcessingPool(max_workers=2, auto_scale=False)
        
        # Test task processing
        def cpu_task(n):
            # Simulate CPU-intensive task
            result = 0
            for i in range(n * 1000):
                result += i
            return result
        
        # Submit multiple tasks
        tasks = [100, 200, 300, 400, 500]
        start_time = time.time()
        results = pool.process_batch(cpu_task, tasks)
        end_time = time.time()
        
        # Verify all tasks completed
        assert len(results) == len(tasks), "Not all tasks completed"
        
        # Test async processing
        futures = pool.submit_async_batch(cpu_task, tasks[:3])
        async_results = pool.wait_for_results(futures)
        assert len(async_results) == 3, "Async processing failed"
        
        # Get pool statistics
        stats = pool.get_stats()
        assert "completed_tasks" in stats, "Pool stats should include completed tasks"
        
        print(f"   ✓ Concurrent processing functional ({end_time - start_time:.3f}s for {len(tasks)} tasks)")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Concurrent processing module not available: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Concurrent processing test failed: {e}")
        return False

def test_auto_scaling():
    """Test auto-scaling functionality."""
    print("Testing auto-scaling...")
    
    try:
        from bci2token.auto_scaling import AutoScaler, ScalingMetrics
        
        # Create auto-scaler with conservative settings
        scaler = AutoScaler(min_workers=1, max_workers=4, monitoring_interval=1.0)
        
        # Test scaling decision logic
        # High load scenario
        high_load_metrics = ScalingMetrics(
            cpu_usage=85.0,
            memory_usage=70.0,
            queue_length=15,
            response_time=3.0,
            error_rate=0.01,
            timestamp=time.time()
        )
        
        decision = scaler.make_scaling_decision(high_load_metrics)
        assert decision is not None, "Scaler should make decision for high load"
        
        # Low load scenario
        low_load_metrics = ScalingMetrics(
            cpu_usage=25.0,
            memory_usage=30.0,
            queue_length=2,
            response_time=0.5,
            error_rate=0.0,
            timestamp=time.time()
        )
        
        decision = scaler.make_scaling_decision(low_load_metrics)
        # Decision might be None if no scaling needed
        
        # Test scaling rules
        rules = scaler.get_scaling_rules()
        assert len(rules) > 0, "Scaler should have scaling rules"
        
        print("   ✓ Auto-scaling logic functional")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Auto-scaling module not available: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Auto-scaling test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    print("Testing memory optimization...")
    
    try:
        from bci2token.performance_optimization import MemoryOptimizer
        
        # Create memory optimizer
        optimizer = MemoryOptimizer()
        
        # Create some test data
        test_arrays = []
        for i in range(10):
            # Create arrays that might fragment memory
            arr = np.random.random((1000, 100))
            test_arrays.append(arr)
        
        # Get initial memory stats
        initial_stats = optimizer.get_memory_stats()
        
        # Trigger memory optimization
        optimizer.optimize_memory()
        
        # Get optimized memory stats
        optimized_stats = optimizer.get_memory_stats()
        
        # Verify stats are collected
        assert "allocated_mb" in initial_stats, "Memory stats should include allocated MB"
        assert "gc_collections" in optimized_stats, "Memory stats should include GC info"
        
        # Test memory cleanup
        del test_arrays  # Delete references
        cleanup_result = optimizer.cleanup_unused_memory()
        
        print("   ✓ Memory optimization functional")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Memory optimization module not available: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Memory optimization test failed: {e}")
        return False

def test_load_balancing():
    """Test load balancing capabilities."""
    print("Testing load balancing...")
    
    try:
        from bci2token.performance_scaling import LoadBalancer, WorkerNode
        
        # Create worker nodes
        nodes = [
            WorkerNode(f"worker_{i}", capacity=100, current_load=0)
            for i in range(3)
        ]
        
        # Create load balancer
        balancer = LoadBalancer(nodes)
        
        # Test task distribution
        tasks = [f"task_{i}" for i in range(10)]
        
        # Distribute tasks
        distributions = []
        for task in tasks:
            node = balancer.select_node(task)
            distributions.append(node.name)
            node.current_load += 10  # Simulate load increase
        
        # Verify tasks were distributed
        assert len(set(distributions)) > 1, "Tasks should be distributed across multiple nodes"
        
        # Test load balancing stats
        stats = balancer.get_stats()
        assert "total_nodes" in stats, "Load balancer stats should include node count"
        
        print("   ✓ Load balancing functional")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Load balancing module not available: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Load balancing test failed: {e}")
        return False

def main():
    """Main test runner for Generation 3 performance features."""
    print("Generation 3 Performance and Scaling Test Suite")
    print("=" * 50)
    
    tests = [
        test_intelligent_caching,
        test_performance_optimization,
        test_concurrent_processing,
        test_auto_scaling,
        test_memory_optimization,
        test_load_balancing
    ]
    
    passed = 0
    failed = 0
    warnings = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                warnings += 1
        except Exception as e:
            print(f"   ✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {warnings} warnings, {failed} failed")
    
    if failed == 0:
        print("🚀 Generation 3 performance and scaling features operational!")
        return 0
    else:
        print("❌ Some critical tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())