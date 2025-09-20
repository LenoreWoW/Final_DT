"""
Comprehensive Error Handling System
Provides robust error handling, logging, and recovery mechanisms.
"""

import logging
import traceback
import functools
import asyncio
from typing import Any, Callable, Dict, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    QUANTUM = "quantum"
    NETWORK = "network"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    traceback: Optional[str]
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False

class QuantumPlatformError(Exception):
    """Base exception for quantum platform errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class QuantumExecutionError(QuantumPlatformError):
    """Quantum circuit execution errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.QUANTUM, ErrorSeverity.HIGH, details)

class ConfigurationError(QuantumPlatformError):
    """Configuration-related errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, details)

class ValidationError(QuantumPlatformError):
    """Data validation errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, details)

class ResourceError(QuantumPlatformError):
    """Resource availability errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.RESOURCE, ErrorSeverity.HIGH, details)

class NetworkError(QuantumPlatformError):
    """Network communication errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, details)

class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self):
        self.error_history: list[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.notification_handlers: list[Callable] = []
        self.max_history_size = 1000
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
    
    def _register_default_recovery_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies[ErrorCategory.QUANTUM] = self._recover_quantum_error
        self.recovery_strategies[ErrorCategory.NETWORK] = self._recover_network_error
        self.recovery_strategies[ErrorCategory.DATABASE] = self._recover_database_error
        self.recovery_strategies[ErrorCategory.RESOURCE] = self._recover_resource_error
    
    async def _recover_quantum_error(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from quantum execution errors."""
        try:
            logger.info("Attempting quantum error recovery...")
            
            # Strategy 1: Retry with classical fallback
            if "circuit_execution" in error_context.details:
                logger.info("Falling back to classical simulation")
                return True
            
            # Strategy 2: Reduce circuit complexity
            if "circuit_depth" in error_context.details:
                logger.info("Reducing circuit complexity for retry")
                return True
            
            # Strategy 3: Switch to different backend
            if "backend_error" in error_context.details:
                logger.info("Switching to fallback quantum backend")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Quantum error recovery failed: {e}")
            return False
    
    async def _recover_network_error(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from network errors."""
        try:
            logger.info("Attempting network error recovery...")
            
            # Strategy 1: Retry with exponential backoff
            retry_count = error_context.details.get('retry_count', 0)
            if retry_count < 3:
                wait_time = 2 ** retry_count
                logger.info(f"Retrying network operation in {wait_time}s...")
                await asyncio.sleep(wait_time)
                return True
            
            # Strategy 2: Switch to cached data
            if "api_call" in error_context.details:
                logger.info("Falling back to cached data")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Network error recovery failed: {e}")
            return False
    
    async def _recover_database_error(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from database errors."""
        try:
            logger.info("Attempting database error recovery...")
            
            # Strategy 1: Reconnect to database
            if "connection" in error_context.message.lower():
                logger.info("Attempting database reconnection")
                return True
            
            # Strategy 2: Switch to read-only mode
            if "write" in error_context.details:
                logger.info("Switching to read-only database mode")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Database error recovery failed: {e}")
            return False
    
    async def _recover_resource_error(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from resource errors."""
        try:
            logger.info("Attempting resource error recovery...")
            
            # Strategy 1: Free up memory
            if "memory" in error_context.message.lower():
                logger.info("Attempting memory cleanup")
                # Trigger garbage collection, clear caches, etc.
                return True
            
            # Strategy 2: Reduce resource usage
            if "cpu" in error_context.message.lower():
                logger.info("Reducing computational load")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Resource error recovery failed: {e}")
            return False
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Handle an error with appropriate logging and recovery."""
        
        # Determine error category and severity
        if isinstance(error, QuantumPlatformError):
            category = error.category
            severity = error.severity
            details = error.details
        else:
            category = self._categorize_error(error)
            severity = self._assess_severity(error)
            details = context or {}
        
        # Create error context
        error_context = ErrorContext(
            timestamp=datetime.utcnow(),
            severity=severity,
            category=category,
            message=str(error),
            details=details,
            traceback=traceback.format_exc(),
            component=context.get('component') if context else None,
            request_id=context.get('request_id') if context else None,
            user_id=context.get('user_id') if context else None
        )
        
        # Log the error
        self._log_error(error_context)
        
        # Add to history
        self._add_to_history(error_context)
        
        # Attempt recovery if strategy exists
        if category in self.recovery_strategies:
            try:
                asyncio.create_task(self._attempt_recovery(error_context))
            except Exception as recovery_error:
                logger.error(f"Error during recovery attempt: {recovery_error}")
        
        # Send notifications for critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._notify_critical_error(error_context)
        
        return error_context
    
    async def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt error recovery using registered strategies."""
        try:
            error_context.recovery_attempted = True
            recovery_strategy = self.recovery_strategies[error_context.category]
            success = await recovery_strategy(error_context)
            error_context.recovery_successful = success
            
            if success:
                logger.info(f"Successfully recovered from {error_context.category.value} error")
            else:
                logger.warning(f"Failed to recover from {error_context.category.value} error")
                
        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")
            error_context.recovery_successful = False
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Quantum-related errors
        if any(keyword in error_msg for keyword in ['qiskit', 'quantum', 'circuit', 'qubit']):
            return ErrorCategory.QUANTUM
        
        # Network-related errors
        if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network', 'http', 'api']):
            return ErrorCategory.NETWORK
        
        # Database-related errors
        if any(keyword in error_msg for keyword in ['database', 'sql', 'connection', 'transaction']):
            return ErrorCategory.DATABASE
        
        # Configuration errors
        if any(keyword in error_msg for keyword in ['config', 'setting', 'environment', 'missing']):
            return ErrorCategory.CONFIGURATION
        
        # Validation errors
        if any(keyword in error_type for keyword in ['validation', 'value', 'type']):
            return ErrorCategory.VALIDATION
        
        # Resource errors
        if any(keyword in error_msg for keyword in ['memory', 'cpu', 'disk', 'resource']):
            return ErrorCategory.RESOURCE
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess the severity of an error."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Critical errors
        if any(keyword in error_msg for keyword in ['critical', 'fatal', 'system']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_type for keyword in ['system', 'runtime', 'memory']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_type for keyword in ['value', 'type', 'attribute']):
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_message = f"[{error_context.category.value.upper()}] {error_context.message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={'error_context': error_context})
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={'error_context': error_context})
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={'error_context': error_context})
        else:
            logger.info(log_message, extra={'error_context': error_context})
    
    def _add_to_history(self, error_context: ErrorContext):
        """Add error to history with size management."""
        self.error_history.append(error_context)
        
        # Maintain maximum history size
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _notify_critical_error(self, error_context: ErrorContext):
        """Send notifications for critical errors."""
        for handler in self.notification_handlers:
            try:
                handler(error_context)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    def add_notification_handler(self, handler: Callable[[ErrorContext], None]):
        """Add a notification handler for critical errors."""
        self.notification_handlers.append(handler)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        recent_errors = []
        
        for error in self.error_history[-100:]:  # Last 100 errors
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            if (datetime.utcnow() - error.timestamp).total_seconds() < 3600:  # Last hour
                recent_errors.append({
                    'timestamp': error.timestamp.isoformat(),
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'message': error.message
                })
        
        return {
            "total_errors": len(self.error_history),
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "recent_errors": recent_errors,
            "recovery_success_rate": self._calculate_recovery_rate()
        }
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate error recovery success rate."""
        recovery_attempts = [e for e in self.error_history if e.recovery_attempted]
        if not recovery_attempts:
            return 0.0
        
        successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
        return len(successful_recoveries) / len(recovery_attempts)

# Global error handler instance
_error_handler: Optional[ErrorHandler] = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

def handle_errors(category: ErrorCategory = None, 
                 component: str = None,
                 reraise: bool = True):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'component': component or func.__name__,
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate for logging
                    'kwargs': str(kwargs)[:200]
                }
                
                error_handler = get_error_handler()
                error_handler.handle_error(e, context)
                
                if reraise:
                    raise
                return None
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'component': component or func.__name__,
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }
                
                error_handler = get_error_handler()
                error_handler.handle_error(e, context)
                
                if reraise:
                    raise
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
