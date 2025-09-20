"""
Performance Profiler for Quantum Trail.
Comprehensive profiling and performance optimization tools.
"""

import time
import psutil
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import functools
import tracemalloc
import linecache
import gc
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class ProfileResult:
    """Result of a performance profiling session."""
    function_name: str
    execution_time: float
    cpu_usage_start: float
    cpu_usage_end: float
    memory_usage_start: float
    memory_usage_end: float
    memory_peak: float
    call_count: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_usage_percent: float
    network_io: Dict[str, Any]
    process_count: int
    load_average: List[float]

class PerformanceProfiler:
    """Main performance profiler class."""
    
    def __init__(self):
        self.profile_results: Dict[str, List[ProfileResult]] = defaultdict(list)
        self.system_metrics: deque = deque(maxlen=1000)
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.system_monitoring_interval = 1.0  # seconds
        self.memory_monitoring_enabled = True
        self.detailed_tracing_enabled = False
        
        # Performance thresholds
        self.performance_thresholds = {
            'execution_time_warning': 1.0,  # seconds
            'execution_time_critical': 5.0,  # seconds
            'memory_usage_warning': 100,     # MB
            'memory_usage_critical': 500,    # MB
            'cpu_usage_warning': 80,         # percent
            'cpu_usage_critical': 95         # percent
        }
        
        logger.info("Performance profiler initialized")
    
    def start_system_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started system performance monitoring")
    
    def stop_system_monitoring(self):
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped system performance monitoring")
    
    def _system_monitor_loop(self):
        """Background loop for system monitoring."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Check for performance issues
                self._check_performance_thresholds(metrics)
                
                time.sleep(self.system_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(5.0)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()._asdict()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available / (1024 * 1024),  # MB
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_io=network,
                process_count=process_count,
                load_average=load_avg
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=0,
                memory_percent=0,
                memory_available=0,
                disk_usage_percent=0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def _check_performance_thresholds(self, metrics: SystemMetrics):
        """Check metrics against performance thresholds."""
        issues = []
        
        # CPU usage
        if metrics.cpu_percent > self.performance_thresholds['cpu_usage_critical']:
            issues.append(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.performance_thresholds['cpu_usage_warning']:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory usage
        if metrics.memory_percent > 95:
            issues.append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > 85:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Load average (Unix)
        if len(metrics.load_average) > 0 and metrics.load_average[0] > psutil.cpu_count() * 2:
            issues.append(f"High system load: {metrics.load_average[0]:.2f}")
        
        if issues:
            logger.warning("Performance issues detected", issues=issues)
    
    def profile_function(self, func: Callable = None, *, name: str = None, 
                        track_memory: bool = True, track_cpu: bool = True):
        """Decorator to profile function performance."""
        def decorator(f):
            @functools.wraps(f)
            def sync_wrapper(*args, **kwargs):
                return self._profile_sync_function(f, name or f.__name__, track_memory, track_cpu, *args, **kwargs)
            
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_function(f, name or f.__name__, track_memory, track_cpu, *args, **kwargs)
            
            if asyncio.iscoroutinefunction(f):
                return async_wrapper
            else:
                return sync_wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _profile_sync_function(self, func: Callable, name: str, track_memory: bool, track_cpu: bool, *args, **kwargs):
        """Profile a synchronous function."""
        start_time = time.time()
        
        # Get initial metrics
        process = psutil.Process()
        cpu_start = process.cpu_percent() if track_cpu else 0
        memory_start = process.memory_info().rss / (1024 * 1024) if track_memory else 0  # MB
        
        # Start memory tracing if enabled
        if track_memory and self.memory_monitoring_enabled:
            tracemalloc.start()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Get final metrics
            execution_time = time.time() - start_time
            cpu_end = process.cpu_percent() if track_cpu else 0
            memory_end = process.memory_info().rss / (1024 * 1024) if track_memory else 0  # MB
            
            # Get memory peak if tracing was enabled
            memory_peak = memory_end
            if track_memory and self.memory_monitoring_enabled and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = peak / (1024 * 1024)  # MB
                tracemalloc.stop()
            
            # Create profile result
            profile_result = ProfileResult(
                function_name=name,
                execution_time=execution_time,
                cpu_usage_start=cpu_start,
                cpu_usage_end=cpu_end,
                memory_usage_start=memory_start,
                memory_usage_end=memory_end,
                memory_peak=memory_peak,
                metadata={'function_type': 'sync'}
            )
            
            # Store result
            self._store_profile_result(profile_result)
            
            # Check for performance issues
            self._check_function_performance(profile_result)
            
            return result
            
        except Exception as e:
            # Clean up memory tracing on error
            if track_memory and tracemalloc.is_tracing():
                tracemalloc.stop()
            raise
    
    async def _profile_async_function(self, func: Callable, name: str, track_memory: bool, track_cpu: bool, *args, **kwargs):
        """Profile an asynchronous function."""
        start_time = time.time()
        
        # Get initial metrics
        process = psutil.Process()
        cpu_start = process.cpu_percent() if track_cpu else 0
        memory_start = process.memory_info().rss / (1024 * 1024) if track_memory else 0  # MB
        
        # Start memory tracing if enabled
        if track_memory and self.memory_monitoring_enabled:
            tracemalloc.start()
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Get final metrics
            execution_time = time.time() - start_time
            cpu_end = process.cpu_percent() if track_cpu else 0
            memory_end = process.memory_info().rss / (1024 * 1024) if track_memory else 0  # MB
            
            # Get memory peak if tracing was enabled
            memory_peak = memory_end
            if track_memory and self.memory_monitoring_enabled and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = peak / (1024 * 1024)  # MB
                tracemalloc.stop()
            
            # Create profile result
            profile_result = ProfileResult(
                function_name=name,
                execution_time=execution_time,
                cpu_usage_start=cpu_start,
                cpu_usage_end=cpu_end,
                memory_usage_start=memory_start,
                memory_usage_end=memory_end,
                memory_peak=memory_peak,
                metadata={'function_type': 'async'}
            )
            
            # Store result
            self._store_profile_result(profile_result)
            
            # Check for performance issues
            self._check_function_performance(profile_result)
            
            return result
            
        except Exception as e:
            # Clean up memory tracing on error
            if track_memory and tracemalloc.is_tracing():
                tracemalloc.stop()
            raise
    
    def _store_profile_result(self, result: ProfileResult):
        """Store a profile result."""
        self.profile_results[result.function_name].append(result)
        
        # Keep only recent results (last 1000 per function)
        if len(self.profile_results[result.function_name]) > 1000:
            self.profile_results[result.function_name] = self.profile_results[result.function_name][-1000:]
    
    def _check_function_performance(self, result: ProfileResult):
        """Check function performance against thresholds."""
        issues = []
        
        # Execution time
        if result.execution_time > self.performance_thresholds['execution_time_critical']:
            issues.append(f"Critical execution time: {result.execution_time:.3f}s")
        elif result.execution_time > self.performance_thresholds['execution_time_warning']:
            issues.append(f"Slow execution time: {result.execution_time:.3f}s")
        
        # Memory usage
        memory_delta = result.memory_usage_end - result.memory_usage_start
        if memory_delta > self.performance_thresholds['memory_usage_critical']:
            issues.append(f"Critical memory usage: {memory_delta:.1f}MB")
        elif memory_delta > self.performance_thresholds['memory_usage_warning']:
            issues.append(f"High memory usage: {memory_delta:.1f}MB")
        
        if issues:
            logger.warning(f"Performance issues in {result.function_name}", issues=issues)
    
    @contextmanager
    def profile_context(self, name: str, track_memory: bool = True, track_cpu: bool = True):
        """Context manager for profiling code blocks."""
        start_time = time.time()
        
        # Get initial metrics
        process = psutil.Process()
        cpu_start = process.cpu_percent() if track_cpu else 0
        memory_start = process.memory_info().rss / (1024 * 1024) if track_memory else 0  # MB
        
        # Start memory tracing if enabled
        if track_memory and self.memory_monitoring_enabled:
            tracemalloc.start()
        
        try:
            yield
            
        finally:
            # Get final metrics
            execution_time = time.time() - start_time
            cpu_end = process.cpu_percent() if track_cpu else 0
            memory_end = process.memory_info().rss / (1024 * 1024) if track_memory else 0  # MB
            
            # Get memory peak if tracing was enabled
            memory_peak = memory_end
            if track_memory and self.memory_monitoring_enabled and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = peak / (1024 * 1024)  # MB
                tracemalloc.stop()
            
            # Create profile result
            profile_result = ProfileResult(
                function_name=name,
                execution_time=execution_time,
                cpu_usage_start=cpu_start,
                cpu_usage_end=cpu_end,
                memory_usage_start=memory_start,
                memory_usage_end=memory_end,
                memory_peak=memory_peak,
                metadata={'function_type': 'context'}
            )
            
            # Store result
            self._store_profile_result(profile_result)
            
            # Check for performance issues
            self._check_function_performance(profile_result)
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistical analysis of a function's performance."""
        if function_name not in self.profile_results:
            return {}
        
        results = self.profile_results[function_name]
        
        if not results:
            return {}
        
        execution_times = [r.execution_time for r in results]
        memory_deltas = [r.memory_usage_end - r.memory_usage_start for r in results]
        
        import statistics
        
        stats = {
            'call_count': len(results),
            'execution_time': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            'memory_usage': {
                'mean': statistics.mean(memory_deltas),
                'median': statistics.median(memory_deltas),
                'min': min(memory_deltas),
                'max': max(memory_deltas),
                'std': statistics.stdev(memory_deltas) if len(memory_deltas) > 1 else 0
            },
            'last_execution': results[-1].timestamp.isoformat(),
            'performance_issues': self._analyze_performance_issues(results)
        }
        
        return stats
    
    def _analyze_performance_issues(self, results: List[ProfileResult]) -> Dict[str, Any]:
        """Analyze performance issues from profile results."""
        issues = {
            'slow_executions': 0,
            'memory_leaks': 0,
            'performance_degradation': False,
            'recommendations': []
        }
        
        # Count slow executions
        for result in results:
            if result.execution_time > self.performance_thresholds['execution_time_warning']:
                issues['slow_executions'] += 1
            
            memory_delta = result.memory_usage_end - result.memory_usage_start
            if memory_delta > self.performance_thresholds['memory_usage_warning']:
                issues['memory_leaks'] += 1
        
        # Check for performance degradation (comparing first 10% vs last 10% of results)
        if len(results) >= 20:
            early_results = results[:len(results)//10]
            recent_results = results[-len(results)//10:]
            
            early_avg = sum(r.execution_time for r in early_results) / len(early_results)
            recent_avg = sum(r.execution_time for r in recent_results) / len(recent_results)
            
            if recent_avg > early_avg * 1.2:  # 20% slower
                issues['performance_degradation'] = True
        
        # Generate recommendations
        if issues['slow_executions'] > len(results) * 0.1:  # More than 10% slow
            issues['recommendations'].append("Consider optimizing algorithm or caching results")
        
        if issues['memory_leaks'] > 0:
            issues['recommendations'].append("Check for memory leaks or excessive memory allocation")
        
        if issues['performance_degradation']:
            issues['recommendations'].append("Performance has degraded over time - investigate recent changes")
        
        return issues
    
    def get_system_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get system performance statistics."""
        if not self.system_metrics:
            return {}
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            filtered_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        else:
            filtered_metrics = list(self.system_metrics)
        
        if not filtered_metrics:
            return {}
        
        cpu_usage = [m.cpu_percent for m in filtered_metrics]
        memory_usage = [m.memory_percent for m in filtered_metrics]
        
        import statistics
        
        stats = {
            'time_period': {
                'start': filtered_metrics[0].timestamp.isoformat(),
                'end': filtered_metrics[-1].timestamp.isoformat(),
                'duration_minutes': len(filtered_metrics) * self.system_monitoring_interval / 60
            },
            'cpu_usage': {
                'mean': statistics.mean(cpu_usage),
                'median': statistics.median(cpu_usage),
                'min': min(cpu_usage),
                'max': max(cpu_usage),
                'std': statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0
            },
            'memory_usage': {
                'mean': statistics.mean(memory_usage),
                'median': statistics.median(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage),
                'std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
            },
            'current_metrics': {
                'cpu_percent': filtered_metrics[-1].cpu_percent,
                'memory_percent': filtered_metrics[-1].memory_percent,
                'memory_available_mb': filtered_metrics[-1].memory_available,
                'disk_usage_percent': filtered_metrics[-1].disk_usage_percent,
                'process_count': filtered_metrics[-1].process_count,
                'load_average': filtered_metrics[-1].load_average
            }
        }
        
        return stats
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'system_stats': self.get_system_stats(timedelta(hours=1)),
            'function_stats': {},
            'top_slow_functions': [],
            'top_memory_consumers': [],
            'recommendations': []
        }
        
        # Get stats for all functions
        for func_name in self.profile_results:
            report['function_stats'][func_name] = self.get_function_stats(func_name)
        
        # Find top slow functions
        slow_functions = []
        for func_name, stats in report['function_stats'].items():
            if stats and 'execution_time' in stats:
                slow_functions.append((func_name, stats['execution_time']['mean']))
        
        slow_functions.sort(key=lambda x: x[1], reverse=True)
        report['top_slow_functions'] = slow_functions[:10]
        
        # Find top memory consumers
        memory_functions = []
        for func_name, stats in report['function_stats'].items():
            if stats and 'memory_usage' in stats:
                memory_functions.append((func_name, stats['memory_usage']['mean']))
        
        memory_functions.sort(key=lambda x: x[1], reverse=True)
        report['top_memory_consumers'] = memory_functions[:10]
        
        # Generate system recommendations
        system_stats = report['system_stats']
        if system_stats and 'cpu_usage' in system_stats:
            if system_stats['cpu_usage']['mean'] > 80:
                report['recommendations'].append("High average CPU usage - consider scaling or optimizing")
            
            if system_stats['memory_usage']['mean'] > 85:
                report['recommendations'].append("High memory usage - monitor for memory leaks")
        
        return report
    
    def force_garbage_collection(self):
        """Force garbage collection and return statistics."""
        gc_stats_before = {
            'collected': gc.get_count(),
            'objects': len(gc.get_objects())
        }
        
        # Force collection
        collected = gc.collect()
        
        gc_stats_after = {
            'collected': gc.get_count(),
            'objects': len(gc.get_objects())
        }
        
        logger.info("Forced garbage collection", 
                   objects_freed=collected,
                   objects_before=gc_stats_before['objects'],
                   objects_after=gc_stats_after['objects'])
        
        return {
            'objects_freed': collected,
            'objects_before': gc_stats_before['objects'],
            'objects_after': gc_stats_after['objects'],
            'memory_freed_estimate': gc_stats_before['objects'] - gc_stats_after['objects']
        }
    
    def clear_profile_data(self, function_name: Optional[str] = None):
        """Clear profile data for a function or all functions."""
        if function_name:
            if function_name in self.profile_results:
                del self.profile_results[function_name]
                logger.info(f"Cleared profile data for {function_name}")
        else:
            self.profile_results.clear()
            self.system_metrics.clear()
            logger.info("Cleared all profile data")

# Global profiler instance
profiler = PerformanceProfiler()

# Convenience functions
def profile(func: Callable = None, *, name: str = None, track_memory: bool = True, track_cpu: bool = True):
    """Decorator to profile function performance."""
    return profiler.profile_function(func, name=name, track_memory=track_memory, track_cpu=track_cpu)

def profile_context(name: str, track_memory: bool = True, track_cpu: bool = True):
    """Context manager for profiling code blocks."""
    return profiler.profile_context(name, track_memory=track_memory, track_cpu=track_cpu)

def start_monitoring():
    """Start system monitoring."""
    profiler.start_system_monitoring()

def stop_monitoring():
    """Stop system monitoring."""
    profiler.stop_system_monitoring()

def get_performance_report():
    """Get comprehensive performance report."""
    return profiler.generate_performance_report()