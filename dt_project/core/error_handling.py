"""Error handling framework stub."""

import enum, functools, asyncio


class ErrorSeverity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(enum.Enum):
    QUANTUM = "quantum"
    NETWORK = "network"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    DATA = "data"


class QuantumPlatformError(Exception):
    def __init__(self, message="", details=None):
        super().__init__(message)
        self.details = details or {}


class QuantumExecutionError(QuantumPlatformError):
    pass


class ConfigurationError(QuantumPlatformError):
    pass


class ValidationError(QuantumPlatformError):
    pass


class ErrorContext:
    def __init__(self, category, severity, details=None):
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recovery_attempted = False


class ErrorHandler:
    def __init__(self):
        self.error_history = []
        self.max_history_size = 1000
        self._notification_handlers = []
        self.recovery_strategies = {}

    def handle_error(self, error):
        category = self._categorize(error)
        severity = self._assess_severity(error)
        details = {}
        if isinstance(error, QuantumPlatformError):
            details = error.details
        ctx = ErrorContext(category, severity, details)

        # Trim history
        if len(self.error_history) >= self.max_history_size:
            self.error_history = self.error_history[-(self.max_history_size - 1):]
        self.error_history.append(ctx)

        # Recovery
        if category in self.recovery_strategies:
            try:
                self.recovery_strategies[category](error)
                ctx.recovery_attempted = True
            except Exception:
                pass

        # Notify
        for handler in self._notification_handlers:
            try:
                handler(ctx)
            except Exception:
                pass

        return ctx

    def _categorize(self, error):
        msg = str(error).lower()
        if isinstance(error, (QuantumExecutionError,)):
            return ErrorCategory.QUANTUM
        if isinstance(error, ConfigurationError):
            return ErrorCategory.CONFIGURATION
        if isinstance(error, (ValueError,)):
            return ErrorCategory.VALIDATION
        if "qiskit" in msg or "quantum" in msg or "circuit" in msg:
            return ErrorCategory.QUANTUM
        if "timeout" in msg or "connection" in msg or "network" in msg:
            return ErrorCategory.NETWORK
        if "invalid" in msg or "value" in msg:
            return ErrorCategory.VALIDATION
        return ErrorCategory.SYSTEM

    def _assess_severity(self, error):
        msg = str(error).lower()
        if isinstance(error, (QuantumExecutionError, ConfigurationError)):
            return ErrorSeverity.HIGH
        if isinstance(error, RuntimeError):
            return ErrorSeverity.HIGH
        if isinstance(error, ValueError):
            return ErrorSeverity.MEDIUM
        if "critical" in msg:
            return ErrorSeverity.CRITICAL
        return ErrorSeverity.MEDIUM

    def get_error_summary(self):
        cats = {}
        sevs = {}
        for ctx in self.error_history:
            cats[ctx.category.value] = cats.get(ctx.category.value, 0) + 1
            sevs[ctx.severity.value] = sevs.get(ctx.severity.value, 0) + 1
        recovered = sum(1 for c in self.error_history if c.recovery_attempted)
        total = len(self.error_history)
        return {
            "total_errors": total,
            "category_counts": cats,
            "severity_counts": sevs,
            "recovery_success_rate": recovered / total if total else 0.0,
        }

    def add_notification_handler(self, handler):
        self._notification_handlers.append(handler)


_global_handler = None

def get_error_handler():
    global _global_handler
    if _global_handler is None:
        _global_handler = ErrorHandler()
    return _global_handler


def handle_errors(component=None, reraise=True):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    handler = get_error_handler()
                    ctx = handler.handle_error(e)
                    ctx.details["component"] = component
                    if reraise:
                        raise
                    return None
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    handler = get_error_handler()
                    ctx = handler.handle_error(e)
                    ctx.details["component"] = component
                    if reraise:
                        raise
                    return None
            return wrapper
    return decorator
