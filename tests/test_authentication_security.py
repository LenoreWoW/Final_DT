"""
Comprehensive tests for authentication and security systems.
Tests critical security vulnerabilities in decorators.py.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from flask import Flask, request, jsonify, g
import json
import time
from datetime import datetime

from dt_project.web_interface.decorators import (
    rate_limit, validate_json, require_auth, validate_circuit_data,
    validate_quantum_params, sanitize_dict, quantum_circuit_timeout,
    cache_result, log_quantum_operation, _validate_auth_token,
    _get_user_from_token, _sanitize_dict, _sanitize_list
)


class TestAuthenticationSecurity:
    """Test suite for authentication and security mechanisms."""

    def setup_method(self):
        """Set up test environment."""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['SECRET_KEY'] = 'test_secret_key'
        self.client = self.app.test_client()

    def test_validate_auth_token_critical_vulnerability(self):
        """Test the critical authentication vulnerability."""
        # CRITICAL: Any 20+ character string should NOT grant access

        # Test that short tokens are rejected
        assert not _validate_auth_token("short", "user")
        assert not _validate_auth_token("", "user")
        assert not _validate_auth_token("a" * 19, "user")

        # VULNERABILITY: 20+ character strings grant access
        assert _validate_auth_token("a" * 20, "user")  # This should FAIL in production
        assert _validate_auth_token("random_long_string_here", "user")

        # CRITICAL VULNERABILITY: Admin access with simple prefix
        assert _validate_auth_token("admin_" + "a" * 15, "admin")  # Trivial admin access
        assert not _validate_auth_token("user_" + "a" * 15, "admin")  # Good: rejects non-admin

        # Test edge cases
        assert _validate_auth_token("admin_minimum_length", "admin")
        assert not _validate_auth_token("admin_short", "admin")

    def test_get_user_from_token_hardcoded_data(self):
        """Test hardcoded user data vulnerability."""
        # CRITICAL: Same user returned for ANY token

        token1 = "any_token_here_that_is_long_enough"
        token2 = "completely_different_token_value"
        token3 = "admin_token_with_different_prefix"

        user1 = _get_user_from_token(token1)
        user2 = _get_user_from_token(token2)
        user3 = _get_user_from_token(token3)

        # VULNERABILITY: All tokens return identical user data
        assert user1 == user2 == user3
        assert user1['user_id'] == 'mock_user_123'
        assert user1['username'] == 'test_user'
        assert user1['permissions'] == ['read', 'write']

    def test_require_auth_decorator_bypass(self):
        """Test authentication decorator bypass vulnerability."""

        @self.app.route('/protected')
        @require_auth('user')
        def protected_route():
            return jsonify({'message': 'success'})

        with self.app.test_request_context():
            # Test without authorization header
            response = self.client.get('/protected')
            assert response.status_code == 401

            # Test with invalid short token
            headers = {'Authorization': 'Bearer short'}
            response = self.client.get('/protected', headers=headers)
            assert response.status_code == 401

            # VULNERABILITY: Any long token grants access
            headers = {'Authorization': 'Bearer this_is_a_very_long_token_that_should_not_work'}
            response = self.client.get('/protected', headers=headers)
            assert response.status_code == 200  # This is the vulnerability!

    def test_sanitize_dict_xss_vulnerability(self):
        """Test input sanitization for XSS vulnerabilities."""

        # Test basic XSS vectors
        malicious_input = {
            'script_tag': '<script>alert("XSS")</script>',
            'img_onerror': '<img src="x" onerror="alert(1)">',
            'iframe': '<iframe src="javascript:alert(1)"></iframe>',
            'svg_xss': '<svg onload="alert(1)">',
            'nested': {
                'deep_xss': '<script>malicious()</script>',
                'list_xss': ['<script>alert(2)</script>', 'safe_value']
            }
        }

        sanitized = sanitize_dict(malicious_input)

        # Verify XSS vectors are handled (current implementation is weak)
        assert '<script>' not in str(sanitized)
        assert 'alert(' not in str(sanitized)

        # Test edge cases
        empty_dict = sanitize_dict({})
        assert empty_dict == {}

        none_values = sanitize_dict({'key': None, 'empty': ''})
        assert none_values['key'] is None
        assert none_values['empty'] == ''

    def test_validate_json_input_validation(self):
        """Test JSON input validation."""

        @self.app.route('/test', methods=['POST'])
        @validate_json(['required_field'], ['optional_field'])
        def test_route():
            return jsonify({'status': 'success'})

        with self.app.test_request_context():
            # Test valid input
            valid_data = {'required_field': 'value', 'optional_field': 'optional'}
            response = self.client.post('/test', json=valid_data)
            assert response.status_code == 200

            # Test missing required field
            invalid_data = {'optional_field': 'optional'}
            response = self.client.post('/test', json=invalid_data)
            assert response.status_code == 400

            # Test non-JSON data
            response = self.client.post('/test', data='not json')
            assert response.status_code == 400

    def test_rate_limiting_bypass(self):
        """Test rate limiting mechanisms."""

        @self.app.route('/limited')
        @rate_limit('quantum_operations')
        def limited_route():
            return jsonify({'message': 'success'})

        with self.app.test_request_context():
            # Test normal requests within limit
            for i in range(5):
                response = self.client.get('/limited')
                assert response.status_code == 200

            # Note: Actual rate limiting behavior depends on configuration
            # and Redis/memory backend - this tests the decorator exists

    def test_validate_circuit_data_quantum_security(self):
        """Test quantum circuit data validation."""

        valid_circuit = {
            'qubits': 4,
            'gates': [
                {'type': 'h', 'qubit': 0},
                {'type': 'cnot', 'control': 0, 'target': 1}
            ],
            'measurements': [{'qubit': 0, 'bit': 0}]
        }

        @self.app.route('/circuit', methods=['POST'])
        @validate_circuit_data
        def circuit_route():
            return jsonify({'status': 'validated'})

        with self.app.test_request_context():
            # Test valid circuit
            response = self.client.post('/circuit', json=valid_circuit)
            assert response.status_code == 200

            # Test invalid circuit structure
            invalid_circuit = {'invalid': 'structure'}
            response = self.client.post('/circuit', json=invalid_circuit)
            assert response.status_code == 400

    def test_quantum_params_validation(self):
        """Test quantum parameter validation."""

        @self.app.route('/quantum', methods=['POST'])
        @validate_quantum_params(['algorithm', 'qubits'])
        def quantum_route():
            return jsonify({'status': 'success'})

        valid_params = {
            'algorithm': 'grover',
            'qubits': 4,
            'iterations': 10
        }

        with self.app.test_request_context():
            # Test valid parameters
            response = self.client.post('/quantum', json=valid_params)
            assert response.status_code == 200

            # Test missing required parameters
            invalid_params = {'algorithm': 'grover'}  # Missing 'qubits'
            response = self.client.post('/quantum', json=invalid_params)
            assert response.status_code == 400

    def test_cache_result_security(self):
        """Test result caching security."""

        @cache_result(expire_time=60)
        def cached_function(data):
            return f"processed_{data}"

        # Test caching works
        result1 = cached_function("test_data")
        result2 = cached_function("test_data")
        assert result1 == result2

        # Test different inputs produce different results
        result3 = cached_function("different_data")
        assert result3 != result1

    def test_log_quantum_operation_audit_trail(self):
        """Test quantum operation logging for audit trails."""

        @log_quantum_operation
        def quantum_operation(circuit_data):
            return {"result": "quantum_computed"}

        with patch('dt_project.web_interface.decorators.logger') as mock_logger:
            result = quantum_operation({"qubits": 4})

            # Verify logging occurred
            assert mock_logger.info.called
            assert result == {"result": "quantum_computed"}

    def test_quantum_circuit_timeout_protection(self):
        """Test timeout protection for quantum circuits."""

        @quantum_circuit_timeout(timeout=1)
        def slow_quantum_operation():
            time.sleep(2)  # Simulate slow operation
            return "completed"

        # This should timeout
        start_time = time.time()
        result = slow_quantum_operation()
        duration = time.time() - start_time

        # Should return within timeout period
        assert duration < 1.5  # Allow some margin

    def test_sanitize_list_nested_xss(self):
        """Test list sanitization for nested XSS."""

        malicious_list = [
            '<script>alert("list_xss")</script>',
            {'nested_dict': '<img src="x" onerror="alert(1)">'},
            ['nested_list', '<iframe src="javascript:alert(1)">']
        ]

        sanitized = _sanitize_list(malicious_list)

        # Verify malicious content is handled
        assert '<script>' not in str(sanitized)
        assert 'javascript:' not in str(sanitized)
        assert 'onerror=' not in str(sanitized)

    def test_security_edge_cases(self):
        """Test edge cases in security functions."""

        # Test with None values
        assert _validate_auth_token(None, "user") == False

        # Test with empty strings
        user_data = _get_user_from_token("")
        assert isinstance(user_data, dict)

        # Test sanitization with None
        sanitized = sanitize_dict(None)
        assert sanitized is None

        # Test with circular references (potential DoS)
        circular_dict = {'key': 'value'}
        circular_dict['self'] = circular_dict

        # This should not cause infinite recursion
        try:
            sanitized = sanitize_dict(circular_dict)
            # If it completes, the function handles circular refs
            assert True
        except RecursionError:
            # If it fails, we've identified a vulnerability
            pytest.fail("Circular reference causes recursion error")

    def test_authorization_level_bypass(self):
        """Test authorization level bypass attempts."""

        # Test admin requirement bypass
        assert not _validate_auth_token("user_very_long_token_here", "admin")
        assert _validate_auth_token("admin_token_here", "admin")

        # Test case sensitivity (potential bypass)
        assert not _validate_auth_token("ADMIN_token_here", "admin")  # Should be case sensitive

        # Test injection attempts in auth level
        assert not _validate_auth_token("admin_token", "admin'; DROP TABLE users; --")

    def test_performance_dos_protection(self):
        """Test protection against performance DoS attacks."""

        # Test large input sanitization (potential DoS)
        large_dict = {}
        for i in range(1000):  # Large but reasonable dict
            large_dict[f'key_{i}'] = f'<script>alert({i})</script>'

        start_time = time.time()
        sanitized = sanitize_dict(large_dict)
        duration = time.time() - start_time

        # Should complete in reasonable time (< 1 second)
        assert duration < 1.0
        assert len(sanitized) == 1000

    def test_token_extraction_security(self):
        """Test token extraction from headers."""

        @self.app.route('/extract_token')
        @require_auth('user')
        def token_route():
            return jsonify({'token_valid': True})

        with self.app.test_request_context():
            # Test different header formats
            valid_formats = [
                'Bearer valid_token_here_long_enough',
                'bearer valid_token_here_long_enough',  # lowercase
            ]

            for auth_header in valid_formats:
                headers = {'Authorization': auth_header}
                response = self.client.get('/extract_token', headers=headers)
                assert response.status_code == 200

            # Test invalid formats
            invalid_formats = [
                'Basic invalid_format',
                'Bearer',  # Missing token
                'valid_token_without_bearer',
                'Bearer short',  # Too short
            ]

            for auth_header in invalid_formats:
                headers = {'Authorization': auth_header}
                response = self.client.get('/extract_token', headers=headers)
                assert response.status_code == 401


class TestSecurityIntegration:
    """Integration tests for security components."""

    def setup_method(self):
        """Set up integration test environment."""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_full_security_chain(self):
        """Test complete security chain integration."""

        @self.app.route('/secure_quantum', methods=['POST'])
        @rate_limit('quantum_operations')
        @require_auth('user')
        @validate_json(['circuit_data'], ['options'])
        @validate_circuit_data
        @log_quantum_operation
        def secure_quantum_endpoint():
            return jsonify({'result': 'quantum_processed'})

        # Test complete valid request
        valid_request = {
            'circuit_data': {
                'qubits': 4,
                'gates': [{'type': 'h', 'qubit': 0}],
                'measurements': [{'qubit': 0, 'bit': 0}]
            },
            'options': {'shots': 1000}
        }

        headers = {'Authorization': 'Bearer valid_token_that_is_long_enough'}

        with self.app.test_request_context():
            response = self.client.post('/secure_quantum',
                                      json=valid_request,
                                      headers=headers)
            assert response.status_code == 200

    def test_security_failure_chain(self):
        """Test security failure propagation."""

        @self.app.route('/fail_chain', methods=['POST'])
        @require_auth('admin')  # Strict requirement
        @validate_json(['admin_action'])
        def admin_endpoint():
            return jsonify({'result': 'admin_action_completed'})

        # Test with user token (should fail auth)
        headers = {'Authorization': 'Bearer user_token_long_enough'}
        request_data = {'admin_action': 'delete_all'}

        with self.app.test_request_context():
            response = self.client.post('/fail_chain',
                                      json=request_data,
                                      headers=headers)
            assert response.status_code == 401  # Auth failure


if __name__ == '__main__':
    pytest.main([__file__, '-v'])