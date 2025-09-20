"""
Test error handling system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from dt_project.core.error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory, QuantumPlatformError,
    QuantumExecutionError, ConfigurationError, ValidationError,
    get_error_handler, handle_errors
)


class TestErrorHandler:
    """Test error handling system."""
    
    def setup_method(self):
        """Set up test."""
        self.error_handler = ErrorHandler()
    
    def test_error_categorization(self):
        """Test automatic error categorization."""
        # Test quantum error
        quantum_error = Exception("Qiskit circuit execution failed")
        context = self.error_handler.handle_error(quantum_error)
        assert context.category == ErrorCategory.QUANTUM
        
        # Test network error
        network_error = Exception("Connection timeout occurred")
        context = self.error_handler.handle_error(network_error)
        assert context.category == ErrorCategory.NETWORK
        
        # Test validation error
        validation_error = ValueError("Invalid parameter value")
        context = self.error_handler.handle_error(validation_error)
        assert context.category == ErrorCategory.VALIDATION
    
    def test_severity_assessment(self):
        """Test automatic severity assessment."""
        # Test critical error
        critical_error = Exception("System critical failure")
        context = self.error_handler.handle_error(critical_error)
        assert context.severity == ErrorSeverity.CRITICAL
        
        # Test high severity error
        runtime_error = RuntimeError("Runtime execution failed")
        context = self.error_handler.handle_error(runtime_error)
        assert context.severity == ErrorSeverity.HIGH
        
        # Test medium severity error
        value_error = ValueError("Invalid value provided")
        context = self.error_handler.handle_error(value_error)
        assert context.severity == ErrorSeverity.MEDIUM
    
    def test_custom_quantum_errors(self):
        """Test custom quantum platform errors."""
        # Test quantum execution error
        quantum_error = QuantumExecutionError("Circuit execution failed", {'qubits': 5})
        context = self.error_handler.handle_error(quantum_error)
        
        assert context.category == ErrorCategory.QUANTUM
        assert context.severity == ErrorSeverity.HIGH
        assert 'qubits' in context.details
        
        # Test configuration error
        config_error = ConfigurationError("Missing API key")
        context = self.error_handler.handle_error(config_error)
        
        assert context.category == ErrorCategory.CONFIGURATION
        assert context.severity == ErrorSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Mock recovery strategy
        mock_recovery = Mock(return_value=True)
        self.error_handler.recovery_strategies[ErrorCategory.QUANTUM] = mock_recovery
        
        # Trigger error
        quantum_error = QuantumExecutionError("Test error")
        context = self.error_handler.handle_error(quantum_error)
        
        # Wait for recovery attempt
        await asyncio.sleep(0.1)
        
        # Check recovery was attempted
        assert context.recovery_attempted == True
    
    def test_error_history(self):
        """Test error history management."""
        # Generate multiple errors
        for i in range(5):
            error = Exception(f"Test error {i}")
            self.error_handler.handle_error(error)
        
        # Check history
        assert len(self.error_handler.error_history) == 5
        
        # Test history size management
        self.error_handler.max_history_size = 3
        for i in range(3):
            error = Exception(f"Additional error {i}")
            self.error_handler.handle_error(error)
        
        assert len(self.error_handler.error_history) <= 3
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Generate errors of different types
        self.error_handler.handle_error(Exception("quantum circuit failed"))
        self.error_handler.handle_error(Exception("network timeout"))
        self.error_handler.handle_error(ValueError("invalid value"))
        
        summary = self.error_handler.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert 'category_counts' in summary
        assert 'severity_counts' in summary
        assert 'recovery_success_rate' in summary
    
    def test_notification_handlers(self):
        """Test critical error notification."""
        # Add mock notification handler
        mock_handler = Mock()
        self.error_handler.add_notification_handler(mock_handler)
        
        # Trigger critical error
        critical_error = Exception("Critical system failure")
        self.error_handler.handle_error(critical_error)
        
        # Check notification was called
        mock_handler.assert_called_once()


class TestErrorDecorator:
    """Test error handling decorator."""
    
    @pytest.mark.asyncio
    async def test_async_error_decorator(self):
        """Test error decorator with async functions."""
        
        @handle_errors(component='test_component')
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await failing_function()
        
        # Check error was handled
        error_handler = get_error_handler()
        assert len(error_handler.error_history) > 0
        assert error_handler.error_history[-1].details['component'] == 'test_component'
    
    def test_sync_error_decorator(self):
        """Test error decorator with sync functions."""
        
        @handle_errors(component='sync_component')
        def failing_sync_function():
            raise RuntimeError("Sync test error")
        
        with pytest.raises(RuntimeError):
            failing_sync_function()
        
        # Check error was handled
        error_handler = get_error_handler()
        assert len(error_handler.error_history) > 0
        assert error_handler.error_history[-1].details['component'] == 'sync_component'
    
    @pytest.mark.asyncio
    async def test_error_decorator_no_reraise(self):
        """Test error decorator without reraising."""
        
        @handle_errors(reraise=False)
        async def failing_function_no_reraise():
            raise ValueError("Test error no reraise")
        
        # Should not raise exception
        result = await failing_function_no_reraise()
        assert result is None
        
        # But error should still be handled
        error_handler = get_error_handler()
        assert len(error_handler.error_history) > 0
