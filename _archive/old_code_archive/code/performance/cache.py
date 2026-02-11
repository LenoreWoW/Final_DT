"""
Advanced Caching System for Quantum Trail.
Provides intelligent caching for quantum computations, simulation results, and data processing.
"""

import time
import json
import pickle
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import threading
import weakref
import structlog

logger = structlog.get_logger(__name__)

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live
    FIFO = "fifo"        # First In, First Out
    ADAPTIVE = "adaptive" # Adaptive policy based on usage patterns

@dataclass
class CacheEntry:
    """Represents a cached item."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    def access(self):
        """Record access to this entry."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0

class InMemoryCache:
    """High-performance in-memory cache with intelligent eviction."""
    
    def __init__(self, max_size_mb: int = 256, max_entries: int = 10000, 
                 default_ttl: Optional[float] = None, policy: CachePolicy = CachePolicy.LRU):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.policy = policy
        
        # Storage
        self.data: Dict[str, CacheEntry] = {}
        if policy == CachePolicy.LRU:
            self.access_order = OrderedDict()
        self.access_frequency = defaultdict(int)
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background cleanup
        self.cleanup_interval = 60.0  # seconds
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_running = False
        
        logger.info(f"Initialized cache: max_size={max_size_mb}MB, max_entries={max_entries}, policy={policy.value}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.data:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
            
            entry = self.data[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
            
            # Record access
            entry.access()
            self.stats.hits += 1
            self.stats.update_hit_rate()
            
            # Update access tracking for eviction policies
            if self.policy == CachePolicy.LRU:
                self.access_order.move_to_end(key)
            elif self.policy == CachePolicy.LFU:
                self.access_frequency[key] += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None, 
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put item in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                # Fallback size estimation
                size_bytes = len(str(value).encode('utf-8'))
            
            # Check if single item exceeds cache capacity
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl,
                metadata=metadata or {}
            )
            
            # Remove existing entry if present
            if key in self.data:
                self._remove_entry(key)
            
            # Ensure capacity
            while (self.stats.size_bytes + size_bytes > self.max_size_bytes or 
                   len(self.data) >= self.max_entries):
                if not self._evict_one():
                    logger.warning("Failed to evict entry for new item")
                    return False
            
            # Add entry
            self.data[key] = entry
            self.stats.size_bytes += size_bytes
            self.stats.entry_count += 1
            
            # Update access tracking
            if self.policy == CachePolicy.LRU:
                self.access_order[key] = True
            elif self.policy == CachePolicy.LFU:
                self.access_frequency[key] = 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self.lock:
            if key in self.data:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all items from cache."""
        with self.lock:
            self.data.clear()
            if self.policy == CachePolicy.LRU:
                self.access_order.clear()
            self.access_frequency.clear()
            self.stats = CacheStats()
    
    def _remove_entry(self, key: str):
        """Remove entry and update statistics."""
        if key in self.data:
            entry = self.data[key]
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            del self.data[key]
            
            # Clean up access tracking
            if self.policy == CachePolicy.LRU and key in self.access_order:
                del self.access_order[key]
            elif self.policy == CachePolicy.LFU and key in self.access_frequency:
                del self.access_frequency[key]
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self.data:
            return False
        
        key_to_evict = None
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            key_to_evict = next(iter(self.access_order))
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            if self.access_frequency:
                key_to_evict = min(self.access_frequency.items(), key=lambda x: x[1])[0]
            else:
                key_to_evict = next(iter(self.data))
        elif self.policy == CachePolicy.FIFO:
            # Remove oldest entry
            oldest_entry = min(self.data.values(), key=lambda x: x.created_at)
            key_to_evict = oldest_entry.key
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.data.items() if v.is_expired()]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                oldest_entry = min(self.data.values(), key=lambda x: x.created_at)
                key_to_evict = oldest_entry.key
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive policy based on access patterns
            key_to_evict = self._adaptive_eviction()
        
        if key_to_evict:
            self._remove_entry(key_to_evict)
            self.stats.evictions += 1
            return True
        
        return False
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction strategy."""
        if not self.data:
            return None
        
        # Score each entry based on multiple factors
        scored_entries = []
        current_time = datetime.utcnow()
        
        for key, entry in self.data.items():
            # Factors: recency, frequency, size, TTL
            recency_score = (current_time - entry.last_accessed).total_seconds()
            frequency_score = 1.0 / (entry.access_count + 1)
            size_score = entry.size_bytes / 1024  # Penalty for large items
            
            # TTL factor
            if entry.ttl_seconds:
                time_since_creation = (current_time - entry.created_at).total_seconds()
                ttl_factor = time_since_creation / entry.ttl_seconds
            else:
                ttl_factor = 0
            
            # Combined score (higher = more likely to evict)
            total_score = recency_score * 0.4 + frequency_score * 0.3 + size_score * 0.2 + ttl_factor * 0.1
            scored_entries.append((key, total_score))
        
        # Return key with highest eviction score
        return max(scored_entries, key=lambda x: x[1])[0]
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        with self.lock:
            expired_keys = [key for key, entry in self.data.items() if entry.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': self.stats.hit_rate,
                'evictions': self.stats.evictions,
                'size_mb': self.stats.size_bytes / (1024 * 1024),
                'size_bytes': self.stats.size_bytes,
                'entry_count': self.stats.entry_count,
                'capacity_mb': self.max_size_bytes / (1024 * 1024),
                'max_entries': self.max_entries,
                'utilization': self.stats.size_bytes / self.max_size_bytes,
                'policy': self.policy.value
            }
    
    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self.cleanup_running:
            return
        
        self.cleanup_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started cache cleanup task")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        self.cleanup_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped cache cleanup task")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.cleanup_running:
            try:
                cleaned = self.cleanup_expired()
                if cleaned > 0:
                    logger.debug(f"Cleaned up {cleaned} expired cache entries")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(5.0)

class QuantumResultCache(InMemoryCache):
    """Specialized cache for quantum computation results."""
    
    def __init__(self, max_size_mb: int = 128, max_entries: int = 5000):
        super().__init__(max_size_mb, max_entries, default_ttl=3600.0, policy=CachePolicy.ADAPTIVE)
    
    def cache_circuit_result(self, circuit_hash: str, backend: str, shots: int, 
                           result: Any, execution_time: float) -> bool:
        """Cache quantum circuit execution result."""
        key = f"circuit:{circuit_hash}:{backend}:{shots}"
        metadata = {
            'result_type': 'quantum_circuit',
            'backend': backend,
            'shots': shots,
            'execution_time': execution_time,
            'cached_at': datetime.utcnow().isoformat()
        }
        return self.put(key, result, metadata=metadata)
    
    def get_circuit_result(self, circuit_hash: str, backend: str, shots: int) -> Optional[Any]:
        """Get cached quantum circuit result."""
        key = f"circuit:{circuit_hash}:{backend}:{shots}"
        return self.get(key)
    
    def cache_optimization_result(self, problem_hash: str, algorithm: str, 
                                result: Any, execution_time: float) -> bool:
        """Cache quantum optimization result."""
        key = f"optimization:{problem_hash}:{algorithm}"
        metadata = {
            'result_type': 'quantum_optimization',
            'algorithm': algorithm,
            'execution_time': execution_time,
            'cached_at': datetime.utcnow().isoformat()
        }
        # Longer TTL for optimization results as they're expensive to compute
        return self.put(key, result, ttl_seconds=7200.0, metadata=metadata)
    
    def get_optimization_result(self, problem_hash: str, algorithm: str) -> Optional[Any]:
        """Get cached quantum optimization result."""
        key = f"optimization:{problem_hash}:{algorithm}"
        return self.get(key)

class MLModelCache(InMemoryCache):
    """Specialized cache for ML model predictions and training results."""
    
    def __init__(self, max_size_mb: int = 512, max_entries: int = 2000):
        super().__init__(max_size_mb, max_entries, default_ttl=1800.0, policy=CachePolicy.LFU)
    
    def cache_prediction(self, model_id: str, input_hash: str, prediction: Any, 
                        confidence: float) -> bool:
        """Cache ML model prediction."""
        key = f"prediction:{model_id}:{input_hash}"
        metadata = {
            'result_type': 'ml_prediction',
            'model_id': model_id,
            'confidence': confidence,
            'cached_at': datetime.utcnow().isoformat()
        }
        # TTL based on confidence
        ttl = 3600.0 if confidence > 0.9 else 1800.0
        return self.put(key, prediction, ttl_seconds=ttl, metadata=metadata)
    
    def get_prediction(self, model_id: str, input_hash: str) -> Optional[Any]:
        """Get cached ML prediction."""
        key = f"prediction:{model_id}:{input_hash}"
        return self.get(key)
    
    def cache_trained_model(self, model_id: str, model_data: Any, 
                           training_metrics: Dict[str, Any]) -> bool:
        """Cache trained ML model."""
        key = f"model:{model_id}"
        metadata = {
            'result_type': 'trained_model',
            'training_metrics': training_metrics,
            'cached_at': datetime.utcnow().isoformat()
        }
        # Long TTL for trained models
        return self.put(key, model_data, ttl_seconds=86400.0, metadata=metadata)
    
    def get_trained_model(self, model_id: str) -> Optional[Any]:
        """Get cached trained model."""
        key = f"model:{model_id}"
        return self.get(key)

def compute_hash(data: Any) -> str:
    """Compute hash for cache key generation."""
    if isinstance(data, dict):
        # Sort dictionary for consistent hashing
        sorted_items = sorted(data.items())
        data_str = json.dumps(sorted_items, sort_keys=True, default=str)
    elif isinstance(data, (list, tuple)):
        data_str = json.dumps(list(data), sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

def cached_quantum_circuit(cache: QuantumResultCache):
    """Decorator for caching quantum circuit execution."""
    def decorator(func):
        def wrapper(circuit_data, backend='simulator', shots=1024, *args, **kwargs):
            # Generate cache key
            circuit_hash = compute_hash({
                'circuit': circuit_data,
                'args': args,
                'kwargs': kwargs
            })
            
            # Try to get from cache
            cached_result = cache.get_circuit_result(circuit_hash, backend, shots)
            if cached_result is not None:
                logger.debug(f"Cache hit for quantum circuit {circuit_hash[:8]}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for quantum circuit {circuit_hash[:8]}")
            start_time = time.time()
            result = func(circuit_data, backend, shots, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result
            cache.cache_circuit_result(circuit_hash, backend, shots, result, execution_time)
            
            return result
        return wrapper
    return decorator

def cached_ml_prediction(cache: MLModelCache):
    """Decorator for caching ML predictions."""
    def decorator(func):
        def wrapper(model_id, input_data, *args, **kwargs):
            # Generate cache key
            input_hash = compute_hash({
                'input_data': input_data,
                'args': args,
                'kwargs': kwargs
            })
            
            # Try to get from cache
            cached_result = cache.get_prediction(model_id, input_hash)
            if cached_result is not None:
                logger.debug(f"Cache hit for ML prediction {model_id}:{input_hash[:8]}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for ML prediction {model_id}:{input_hash[:8]}")
            result = func(model_id, input_data, *args, **kwargs)
            
            # Extract confidence if available
            if isinstance(result, dict) and 'confidence' in result:
                confidence = result['confidence']
            else:
                confidence = 0.8  # Default confidence
            
            # Cache result
            cache.cache_prediction(model_id, input_hash, result, confidence)
            
            return result
        return wrapper
    return decorator

# Global cache instances
quantum_cache = QuantumResultCache()
ml_cache = MLModelCache()
general_cache = InMemoryCache(max_size_mb=64, max_entries=1000)

# Convenience functions
def start_cache_cleanup():
    """Start cleanup tasks for all caches."""
    async def start_all():
        await quantum_cache.start_cleanup_task()
        await ml_cache.start_cleanup_task()
        await general_cache.start_cleanup_task()
    
    asyncio.create_task(start_all())
    logger.info("Started cache cleanup tasks")

def stop_cache_cleanup():
    """Stop cleanup tasks for all caches."""
    async def stop_all():
        await quantum_cache.stop_cleanup_task()
        await ml_cache.stop_cleanup_task()
        await general_cache.stop_cleanup_task()
    
    asyncio.create_task(stop_all())
    logger.info("Stopped cache cleanup tasks")

def get_cache_statistics() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        'quantum_cache': quantum_cache.get_stats(),
        'ml_cache': ml_cache.get_stats(),
        'general_cache': general_cache.get_stats(),
        'total_size_mb': (
            quantum_cache.get_stats()['size_mb'] +
            ml_cache.get_stats()['size_mb'] +
            general_cache.get_stats()['size_mb']
        )
    }