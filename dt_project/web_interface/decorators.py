"""
Flask decorators for security, validation, and monitoring.
"""

import functools
import logging
import time
import asyncio
from typing import Callable, List, Optional
from flask import request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from dt_project.config.secure_config import get_config, validate_input, sanitize_input

logger = logging.getLogger(__name__)

# Initialize rate limiter
config = get_config()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config.get('RATE_LIMITS.DEFAULT', '100 per hour')]
)

def rate_limit(limit_type: str):
    """
    Apply rate limiting based on limit type.
    
    Args:
        limit_type: Type of rate limit ('default', 'api_calls', 'quantum_operations')
    """
    def decorator(f):
        limit_mapping = {
            'default': config.get('RATE_LIMITS.DEFAULT', '100 per hour'),
            'api_calls': config.get('RATE_LIMITS.API_CALLS', '1000 per hour'),
            'quantum_operations': config.get('RATE_LIMITS.QUANTUM_OPERATIONS', '10 per hour')
        }
        
        rate_limit_value = limit_mapping.get(limit_type, limit_mapping['default'])
        
        @limiter.limit(rate_limit_value)
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def validate_json(required_fields: List[str], optional_fields: Optional[List[str]] = None):
    """
    Validate JSON input and sanitize string fields.
    
    Args:
        required_fields: List of required field names
        optional_fields: List of optional field names
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if request is JSON
            if not request.is_json:
                logger.warning("Non-JSON request received", extra={'path': request.path})
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Empty request body'}), 400
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                logger.warning(f"Missing required fields: {missing_fields}")
                return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
            
            # Sanitize all string inputs
            sanitized_data = {}
            all_fields = required_fields + (optional_fields or [])
            
            for field in all_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, str):
                        sanitized_data[field] = sanitize_input(value)
                    elif isinstance(value, dict):
                        sanitized_data[field] = _sanitize_dict(value)
                    elif isinstance(value, list):
                        sanitized_data[field] = _sanitize_list(value)
                    else:
                        sanitized_data[field] = value
            
            # Add sanitized data to request context
            request.validated_data = sanitized_data
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def require_auth(auth_level: str = 'user'):
    """
    Require authentication for endpoint access.
    
    Args:
        auth_level: Required authentication level ('user', 'admin', 'service')
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Get authentication token from header
            auth_header = request.headers.get('Authorization')
            
            if not auth_header or not auth_header.startswith('Bearer '):
                logger.warning("Missing or invalid authentication header")
                return jsonify({'error': 'Authentication required'}), 401
            
            token = auth_header.split(' ')[1]
            
            # Validate token (simplified - in real implementation, use proper JWT validation)
            if not _validate_auth_token(token, auth_level):
                logger.warning(f"Invalid authentication token for level {auth_level}")
                return jsonify({'error': 'Invalid or insufficient permissions'}), 403
            
            # Store user info in request context
            g.user_info = _get_user_from_token(token)
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def log_request(f):
    """Log incoming requests with timing information."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # Log request start
        logger.info("Request started", extra={
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown')[:100]  # Truncate long user agents
        })
        
        try:
            result = f(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful request
            logger.info("Request completed", extra={
                'path': request.path,
                'execution_time': f"{execution_time:.3f}s",
                'status': 'success'
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log failed request
            logger.error("Request failed", extra={
                'path': request.path,
                'execution_time': f"{execution_time:.3f}s",
                'error': str(e)
            })
            
            raise
    
    return decorated_function

def async_route(f):
    """
    Decorator to handle async functions in Flask routes.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async function
            return loop.run_until_complete(f(*args, **kwargs))
            
        except Exception as e:
            logger.error(f"Async route error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return decorated_function

def cache_response(expiry_seconds: int = 300):
    """
    Cache response for specified duration.
    
    Args:
        expiry_seconds: Cache expiry time in seconds
    """
    def decorator(f):
        cache = {}
        
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Create cache key from request data
            cache_key = f"{request.method}:{request.path}:{request.args.to_dict()}"
            
            # Check cache
            if cache_key in cache:
                cached_data, timestamp = cache[cache_key]
                if time.time() - timestamp < expiry_seconds:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_data
                else:
                    # Remove expired entry
                    del cache[cache_key]
            
            # Execute function and cache result
            result = f(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            # Clean up old cache entries periodically
            if len(cache) > 1000:  # Arbitrary limit
                current_time = time.time()
                expired_keys = [
                    key for key, (_, timestamp) in cache.items()
                    if current_time - timestamp >= expiry_seconds
                ]
                for key in expired_keys:
                    del cache[key]
            
            return result
        
        return decorated_function
    return decorator

def validate_content_type(content_types: List[str]):
    """
    Validate request content type.
    
    Args:
        content_types: List of allowed content types
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if request.content_type not in content_types:
                logger.warning(f"Invalid content type: {request.content_type}")
                return jsonify({'error': f'Content type must be one of: {content_types}'}), 400
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def monitor_performance(threshold_seconds: float = 5.0):
    """
    Monitor endpoint performance and log slow requests.
    
    Args:
        threshold_seconds: Threshold above which to log performance warning
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            
            result = f(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            if execution_time > threshold_seconds:
                logger.warning("Slow request detected", extra={
                    'path': request.path,
                    'execution_time': f"{execution_time:.3f}s",
                    'threshold': f"{threshold_seconds}s"
                })
            
            return result
        
        return decorated_function
    return decorator

# Helper functions
def _sanitize_dict(data: dict) -> dict:
    """Recursively sanitize dictionary values."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_input(value)
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[key] = _sanitize_list(value)
        else:
            sanitized[key] = value
    return sanitized

def _sanitize_list(data: list) -> list:
    """Recursively sanitize list values."""
    sanitized = []
    for item in data:
        if isinstance(item, str):
            sanitized.append(sanitize_input(item))
        elif isinstance(item, dict):
            sanitized.append(_sanitize_dict(item))
        elif isinstance(item, list):
            sanitized.append(_sanitize_list(item))
        else:
            sanitized.append(item)
    return sanitized

def _validate_auth_token(token: str, required_level: str) -> bool:
    """
    Validate authentication token (simplified implementation).
    
    In a real application, this would validate JWT tokens,
    check against a user database, verify permissions, etc.
    """
    # Simplified validation - just check token format
    if len(token) < 20:
        return False
    
    # Mock validation based on token prefix
    if required_level == 'admin' and not token.startswith('admin_'):
        return False
    
    return True

def _get_user_from_token(token: str) -> dict:
    """
    Extract user information from authentication token.
    
    In a real application, this would decode JWT and return user data.
    """
    return {
        'user_id': 'mock_user_123',
        'username': 'test_user',
        'permissions': ['read', 'write']
    }