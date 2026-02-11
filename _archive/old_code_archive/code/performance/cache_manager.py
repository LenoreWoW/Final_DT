"""
High-performance cache manager with multiple backends.
"""

import time
import hashlib
import pickle
import logging
from typing import Any, Optional, Union, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class CacheBackend(Enum):
    """Available cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_accessed = time.time()

class CacheBackendInterface(ABC):
    """Abstract interface for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache entry."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 10000, cleanup_interval: int = 300):
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
        self._last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry with LRU update."""
        with self._lock:
            self._cleanup_if_needed()
            
            entry = self._cache.get(key)
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._stats['misses'] += 1
                return None
            
            entry.touch()
            self._stats['hits'] += 1
            return entry
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache entry with automatic eviction."""
        try:
            with self._lock:
                expires_at = None
                if ttl is not None:
                    expires_at = time.time() + ttl
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    expires_at=expires_at
                )
                
                self._cache[key] = entry
                self._stats['sets'] += 1
                
                # Evict if over size limit
                if len(self._cache) > self.max_size:
                    self._evict_lru()
                
                return True
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['deletes'] += 1
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            return True
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': self._stats['hits'] / max(self._stats['hits'] + self._stats['misses'], 1)
            }
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        # Sort by last accessed time and remove oldest 20%
        entries = list(self._cache.items())
        entries.sort(key=lambda x: x[1].last_accessed)
        
        evict_count = max(1, len(entries) // 5)  # Remove 20%
        
        for key, _ in entries[:evict_count]:
            del self._cache[key]
            self._stats['evictions'] += 1
    
    def _cleanup_if_needed(self):
        """Clean up expired entries if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and current_time > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

class RedisCacheBackend(CacheBackendInterface):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "dt_cache:"):
        self.prefix = prefix
        self._redis = None
        self._stats = {'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0}
        
        try:
            import redis
            self._redis = redis.from_url(redis_url)
            # Test connection
            self._redis.ping()
            logger.info("Redis cache backend initialized")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self._redis = None
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from Redis."""
        if not self._redis:
            return None
        
        try:
            redis_key = f"{self.prefix}{key}"
            data = self._redis.get(redis_key)
            
            if data is None:
                self._stats['misses'] += 1
                return None
            
            entry = pickle.loads(data)
            if entry.is_expired():
                self._redis.delete(redis_key)
                self._stats['misses'] += 1
                return None
            
            entry.touch()
            # Update entry in Redis with new access info
            self._redis.setex(redis_key, 
                            int((entry.expires_at or time.time() + 3600) - time.time()), 
                            pickle.dumps(entry))
            
            self._stats['hits'] += 1
            return entry
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache entry in Redis."""
        if not self._redis:
            return False
        
        try:
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at
            )
            
            redis_key = f"{self.prefix}{key}"
            data = pickle.dumps(entry)
            
            if ttl is not None:
                self._redis.setex(redis_key, ttl, data)
            else:
                self._redis.set(redis_key, data)
            
            self._stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry from Redis."""
        if not self._redis:
            return False
        
        try:
            redis_key = f"{self.prefix}{key}"
            result = self._redis.delete(redis_key)
            if result:
                self._stats['deletes'] += 1
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        if not self._redis:
            return False
        
        try:
            pattern = f"{self.prefix}*"
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        redis_info = {}
        if self._redis:
            try:
                redis_info = self._redis.info('memory')
            except:
                pass
        
        return {
            **self._stats,
            'redis_available': self._redis is not None,
            'redis_memory': redis_info.get('used_memory_human', 'unknown'),
            'hit_rate': self._stats['hits'] / max(self._stats['hits'] + self._stats['misses'], 1)
        }

class CacheManager:
    """High-level cache manager with multiple backends and intelligent routing."""
    
    def __init__(self, 
                 primary_backend: CacheBackend = CacheBackend.MEMORY,
                 fallback_backend: Optional[CacheBackend] = None,
                 cache_config: Optional[Dict] = None):
        
        self.config = cache_config or {}
        self.backends = {}
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize backends
        self._initialize_backends(primary_backend, fallback_backend)
        
        self.primary_backend = primary_backend
        self.fallback_backend = fallback_backend
    
    def _initialize_backends(self, primary: CacheBackend, fallback: Optional[CacheBackend]):
        """Initialize cache backends."""
        # Memory backend (always available)
        self.backends[CacheBackend.MEMORY] = MemoryCacheBackend(
            max_size=self.config.get('memory_max_size', 10000),
            cleanup_interval=self.config.get('memory_cleanup_interval', 300)
        )
        
        # Redis backend (if available)
        if CacheBackend.REDIS in [primary, fallback]:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379/0')
            redis_backend = RedisCacheBackend(redis_url)
            if redis_backend._redis:  # Only add if Redis is available
                self.backends[CacheBackend.REDIS] = redis_backend
        
        logger.info(f"Initialized cache backends: {list(self.backends.keys())}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with fallback support."""
        self._stats['total_requests'] += 1
        
        # Generate cache key hash for consistent lookups
        cache_key = self._generate_key(key)
        
        # Try primary backend
        if self.primary_backend in self.backends:
            entry = self.backends[self.primary_backend].get(cache_key)
            if entry:
                self._stats['cache_hits'] += 1
                return entry.value
        
        # Try fallback backend
        if self.fallback_backend and self.fallback_backend in self.backends:
            entry = self.backends[self.fallback_backend].get(cache_key)
            if entry:
                self._stats['cache_hits'] += 1
                # Promote to primary backend
                self.set(key, entry.value, ttl=300)  # 5 minute TTL for promoted entries
                return entry.value
        
        self._stats['cache_misses'] += 1
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache across backends."""
        cache_key = self._generate_key(key)
        success = False
        
        # Set in primary backend
        if self.primary_backend in self.backends:
            success = self.backends[self.primary_backend].set(cache_key, value, ttl)
        
        # Set in fallback backend (for redundancy)
        if self.fallback_backend and self.fallback_backend in self.backends:
            self.backends[self.fallback_backend].set(cache_key, value, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete value from all backends."""
        cache_key = self._generate_key(key)
        success = False
        
        for backend in self.backends.values():
            if backend.delete(cache_key):
                success = True
        
        return success
    
    def clear(self) -> bool:
        """Clear all backends."""
        success = True
        for backend in self.backends.values():
            if not backend.clear():
                success = False
        return success
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        backend_stats = {}
        for backend_type, backend in self.backends.items():
            backend_stats[backend_type.value] = backend.stats()
        
        return {
            'manager_stats': self._stats,
            'hit_rate': self._stats['cache_hits'] / max(self._stats['total_requests'], 1),
            'backends': backend_stats
        }
    
    def _generate_key(self, key: str) -> str:
        """Generate consistent cache key."""
        # Use SHA-256 hash to ensure consistent key length and avoid special characters
        return hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]

# Decorators for caching
def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            result = cache_manager.get(cache_key)
            
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(
            primary_backend=CacheBackend.MEMORY,
            fallback_backend=CacheBackend.REDIS
        )
    return _cache_manager

def configure_cache(config: Dict[str, Any]):
    """Configure global cache manager."""
    global _cache_manager
    primary = CacheBackend(config.get('primary_backend', 'memory'))
    fallback = CacheBackend(config.get('fallback_backend')) if config.get('fallback_backend') else None
    
    _cache_manager = CacheManager(
        primary_backend=primary,
        fallback_backend=fallback,
        cache_config=config
    )