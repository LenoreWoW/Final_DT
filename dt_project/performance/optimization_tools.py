"""
Performance optimization tools and utilities.
"""

import time
import numpy as np
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance measurement result."""
    operation: str
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

class PerformanceProfiler:
    """Advanced performance profiler for identifying bottlenecks."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory if end_memory and start_memory else None
            
            metric = PerformanceMetric(
                operation=operation_name,
                execution_time=execution_time,
                memory_usage=memory_delta,
                success=True
            )
            
            with self._lock:
                self.metrics.append(metric)
            
            if execution_time > 1.0:  # Log slow operations
                logger.warning(f"Slow operation detected: {operation_name} took {execution_time:.2f}s")
                
        except Exception as e:
            execution_time = time.time() - start_time
            metric = PerformanceMetric(
                operation=operation_name,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
            
            with self._lock:
                self.metrics.append(metric)
            
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if not self.metrics:
                return {'message': 'No performance data available'}
            
            # Group by operation
            operation_stats = {}
            for metric in self.metrics:
                op_name = metric.operation
                if op_name not in operation_stats:
                    operation_stats[op_name] = {
                        'count': 0,
                        'total_time': 0,
                        'min_time': float('inf'),
                        'max_time': 0,
                        'avg_time': 0,
                        'success_rate': 0,
                        'total_memory_delta': 0
                    }
                
                stats = operation_stats[op_name]
                stats['count'] += 1
                stats['total_time'] += metric.execution_time
                stats['min_time'] = min(stats['min_time'], metric.execution_time)
                stats['max_time'] = max(stats['max_time'], metric.execution_time)
                
                if metric.success:
                    stats['success_rate'] += 1
                
                if metric.memory_usage:
                    stats['total_memory_delta'] += metric.memory_usage
            
            # Calculate averages and rates
            for stats in operation_stats.values():
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['success_rate'] = stats['success_rate'] / stats['count']
                stats['avg_memory_delta'] = stats['total_memory_delta'] / stats['count']
            
            # Overall stats
            total_operations = len(self.metrics)
            successful_operations = sum(1 for m in self.metrics if m.success)
            
            return {
                'overall': {
                    'total_operations': total_operations,
                    'successful_operations': successful_operations,
                    'success_rate': successful_operations / total_operations,
                    'avg_execution_time': sum(m.execution_time for m in self.metrics) / total_operations
                },
                'by_operation': operation_stats
            }
    
    def clear_stats(self):
        """Clear all performance metrics."""
        with self._lock:
            self.metrics.clear()
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None

class BatchProcessor:
    """Efficient batch processing with parallel execution."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, 
                     items: List[Any], 
                     processor_func: Callable,
                     batch_size: int = 100,
                     progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Process items in batches with parallel execution.
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            batch_size: Size of each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches in parallel
        future_to_batch = {
            self.executor.submit(self._process_batch_chunk, batch, processor_func): batch_idx
            for batch_idx, batch in enumerate(batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
                
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx} processing failed: {e}")
                # Add None results for failed batch
                results.extend([None] * len(batches[batch_idx]))
        
        return results
    
    def _process_batch_chunk(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process a single batch chunk."""
        results = []
        for item in batch:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Item processing failed: {e}")
                results.append(None)
        return results

class GeospatialOptimizer:
    """Optimized geospatial calculations using vectorization."""
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points (cached).
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    @staticmethod
    def vectorized_haversine(lat1_arr: np.ndarray, lon1_arr: np.ndarray,
                           lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
        """
        Vectorized haversine distance calculation for arrays of points.
        
        Args:
            lat1_arr, lon1_arr: Arrays of first points
            lat2_arr, lon2_arr: Arrays of second points
            
        Returns:
            Array of distances in meters
        """
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = np.radians(lat1_arr)
        lat2_rad = np.radians(lat2_arr)
        delta_lat = np.radians(lat2_arr - lat1_arr)
        delta_lon = np.radians(lon2_arr - lon1_arr)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    @staticmethod
    def calculate_route_distances(route_points: List[Dict[str, float]]) -> List[float]:
        """
        Calculate distances between consecutive route points efficiently.
        
        Args:
            route_points: List of points with 'latitude' and 'longitude' keys
            
        Returns:
            List of distances between consecutive points
        """
        if len(route_points) < 2:
            return []
        
        # Convert to numpy arrays for vectorization
        lats = np.array([p['latitude'] for p in route_points])
        lons = np.array([p['longitude'] for p in route_points])
        
        # Calculate distances between consecutive points
        lat1_arr = lats[:-1]
        lon1_arr = lons[:-1]
        lat2_arr = lats[1:]
        lon2_arr = lons[1:]
        
        return GeospatialOptimizer.vectorized_haversine(lat1_arr, lon1_arr, lat2_arr, lon2_arr).tolist()

class DataOptimizer:
    """Data structure and algorithm optimizations."""
    
    @staticmethod
    def optimize_athlete_data(athlete_profiles: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Optimize athlete profile data for fast access and computation.
        
        Args:
            athlete_profiles: List of athlete profile dictionaries
            
        Returns:
            Tuple of (optimized_array, index_mapping)
        """
        if not athlete_profiles:
            return np.array([]), {}
        
        # Extract numeric fields for vectorized operations
        numeric_fields = ['age', 'weight', 'height', 'fitness_level']
        available_fields = [field for field in numeric_fields 
                          if any(field in profile for profile in athlete_profiles)]
        
        # Create structured array
        data = []
        index_mapping = {}
        
        for idx, profile in enumerate(athlete_profiles):
            row = []
            for field in available_fields:
                row.append(profile.get(field, 0.0))
            data.append(row)
            
            # Map athlete name/id to array index
            key = profile.get('name') or profile.get('id') or str(idx)
            index_mapping[key] = idx
        
        optimized_array = np.array(data, dtype=np.float32)
        
        return optimized_array, index_mapping
    
    @staticmethod
    def fast_terrain_lookup(terrain_points: List[Dict], 
                           query_lat: float, 
                           query_lon: float,
                           max_distance: float = 1000.0) -> Optional[Dict]:
        """
        Fast terrain point lookup using spatial indexing.
        
        Args:
            terrain_points: List of terrain point dictionaries
            query_lat, query_lon: Query coordinates
            max_distance: Maximum search distance in meters
            
        Returns:
            Closest terrain point within max_distance or None
        """
        if not terrain_points:
            return None
        
        # Convert to arrays for vectorized distance calculation
        lats = np.array([p['latitude'] for p in terrain_points])
        lons = np.array([p['longitude'] for p in terrain_points])
        
        # Calculate distances to all points
        distances = GeospatialOptimizer.vectorized_haversine(
            np.full_like(lats, query_lat),
            np.full_like(lons, query_lon),
            lats,
            lons
        )
        
        # Find closest point within max distance
        valid_indices = distances <= max_distance
        if not np.any(valid_indices):
            return None
        
        valid_distances = distances[valid_indices]
        min_idx = np.argmin(valid_distances)
        original_idx = np.where(valid_indices)[0][min_idx]
        
        return terrain_points[original_idx]

# Performance monitoring decorators
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            with profiler.profile(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def optimize_for_numpy(func):
    """Decorator to optimize functions that work with numpy arrays."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert lists to numpy arrays if beneficial
        optimized_args = []
        for arg in args:
            if isinstance(arg, list) and len(arg) > 100:
                # Only convert large lists to numpy arrays
                try:
                    optimized_args.append(np.array(arg))
                except (ValueError, TypeError):
                    optimized_args.append(arg)
            else:
                optimized_args.append(arg)
        
        return func(*optimized_args, **kwargs)
    return wrapper

def memory_efficient(max_memory_mb: float = 500):
    """Decorator to ensure memory-efficient execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import gc
            
            # Force garbage collection before execution
            gc.collect()
            
            initial_memory = _get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                final_memory = _get_memory_usage()
                if initial_memory and final_memory:
                    memory_increase = final_memory - initial_memory
                    if memory_increase > max_memory_mb:
                        logger.warning(f"High memory usage detected: {memory_increase:.1f}MB in {func.__name__}")
                
                return result
                
            finally:
                # Force garbage collection after execution
                gc.collect()
        
        return wrapper
    return decorator

# Global performance profiler instance
_performance_profiler = None

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler

def _get_memory_usage() -> Optional[float]:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return None

# Quantum-specific optimizations
class QuantumCircuitOptimizer:
    """Optimizations specific to quantum circuits."""
    
    @staticmethod
    def optimize_circuit_depth(circuit_gates: List[Dict]) -> List[Dict]:
        """
        Optimize quantum circuit depth by reordering gates.
        
        Args:
            circuit_gates: List of gate dictionaries
            
        Returns:
            Optimized gate list
        """
        if not circuit_gates:
            return []
        
        # Group gates that can be parallelized (acting on different qubits)
        optimized_gates = []
        used_qubits = set()
        current_layer = []
        
        for gate in circuit_gates:
            gate_qubits = set(gate.get('qubits', []))
            
            # If gate uses qubits already in current layer, start new layer
            if gate_qubits & used_qubits:
                optimized_gates.extend(current_layer)
                current_layer = [gate]
                used_qubits = gate_qubits
            else:
                current_layer.append(gate)
                used_qubits.update(gate_qubits)
        
        # Add final layer
        optimized_gates.extend(current_layer)
        
        logger.info(f"Circuit optimization: {len(circuit_gates)} -> {len(optimized_gates)} gates")
        return optimized_gates
    
    @staticmethod
    def estimate_execution_time(n_qubits: int, n_gates: int, shots: int, backend_type: str = 'simulator') -> float:
        """
        Estimate quantum circuit execution time.
        
        Args:
            n_qubits: Number of qubits
            n_gates: Number of gates
            shots: Number of measurements
            backend_type: Type of backend ('simulator' or 'hardware')
            
        Returns:
            Estimated execution time in seconds
        """
        if backend_type == 'simulator':
            # Simulator time scales exponentially with qubits, linearly with gates and shots
            base_time = 0.001  # Base time per operation
            qubit_factor = 2 ** min(n_qubits, 20)  # Cap at 20 qubits for estimation
            gate_factor = n_gates
            shot_factor = shots / 1000  # Normalize shots
            
            return base_time * qubit_factor * gate_factor * shot_factor
        
        else:  # hardware
            # Hardware has queue time and execution time
            queue_time = 30.0  # Estimated queue time
            execution_time = 0.1 * n_gates + 0.01 * shots  # Rough hardware timing
            
            return queue_time + execution_time