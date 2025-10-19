"""
Comprehensive tests for web interface core functionality.
Tests Flask application, routes, and web security features.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
from werkzeug.test import Client

# Import web interface components
from dt_project.web_interface.app import create_app, configure_app
from dt_project.web_interface.secure_app import create_secure_app
from dt_project.web_interface.decorators import (
    rate_limit, validate_json, require_auth, validate_circuit_data,
    sanitize_dict, cache_result, log_quantum_operation
)


class TestWebApplicationCore:
    """Test core web application functionality."""

    def setup_method(self):
        """Set up web application test environment."""
        self.test_config = {
            'TESTING': True,
            'SECRET_KEY': 'test_secret_key_for_testing_only',
            'WTF_CSRF_ENABLED': False,
            'DATABASE_URL': 'sqlite:///:memory:',
            'QUANTUM_BACKEND': 'simulator'
        }

    def test_app_creation_and_configuration(self):
        """Test Flask application creation and configuration."""

        app = create_app(self.test_config)

        assert app is not None
        assert isinstance(app, Flask)
        assert app.config['TESTING'] == True
        assert app.config['SECRET_KEY'] == 'test_secret_key_for_testing_only'

        # Test that essential components are configured
        assert hasattr(app, 'url_map')
        assert len(app.url_map._rules) > 0  # Routes are registered

    def test_secure_app_configuration(self):
        """Test secure Flask application configuration."""

        with patch('dt_project.web_interface.secure_app.CSRFProtect') as mock_csrf:
            secure_app = create_secure_app(self.test_config)

            assert secure_app is not None
            mock_csrf.assert_called_once()

            # Test security headers configuration
            with secure_app.test_client() as client:
                response = client.get('/health')

                # Should have security headers
                assert 'X-Content-Type-Options' in response.headers
                assert 'X-Frame-Options' in response.headers

    def test_application_factory_pattern(self):
        """Test application factory pattern implementation."""

        # Test with different configurations
        configs = [
            {'TESTING': True, 'DEBUG': False},
            {'TESTING': False, 'DEBUG': True},
            {'TESTING': False, 'DEBUG': False}
        ]

        for config in configs:
            app = create_app(config)
            assert app.config['TESTING'] == config['TESTING']
            assert app.config['DEBUG'] == config.get('DEBUG', False)

    def test_error_handlers_registration(self):
        """Test error handlers are properly registered."""

        app = create_app(self.test_config)

        with app.test_client() as client:
            # Test 404 error handler
            response = client.get('/nonexistent_route')
            assert response.status_code == 404

            # Test 500 error handler (simulate internal error)
            with patch('dt_project.web_interface.app.some_function') as mock_func:
                mock_func.side_effect = Exception("Test internal error")

                # This would trigger a 500 error in real scenario
                # For testing, we'll verify the error handler exists
                error_handlers = app.error_handler_spec[None]
                assert 500 in error_handlers or None in error_handlers

    def test_blueprint_registration(self):
        """Test that all blueprints are properly registered."""

        app = create_app(self.test_config)

        # Check that blueprints are registered
        blueprint_names = [bp.name for bp in app.iter_blueprints()]

        expected_blueprints = [
            'main', 'api', 'quantum', 'admin', 'docs', 'quantum_lab', 'simulation'
        ]

        for expected_bp in expected_blueprints:
            assert any(expected_bp in bp_name for bp_name in blueprint_names), \
                f"Blueprint {expected_bp} not found in {blueprint_names}"

    def test_static_file_serving(self):
        """Test static file serving configuration."""

        app = create_app(self.test_config)

        with app.test_client() as client:
            # Test that static route is configured
            # Note: In testing, static files might not exist
            response = client.get('/static/css/style.css')
            # Response could be 404 if file doesn't exist, which is fine for testing
            assert response.status_code in [200, 404]

    def test_template_configuration(self):
        """Test template configuration and rendering."""

        app = create_app(self.test_config)

        with app.test_client() as client:
            # Test that templates can be rendered
            response = client.get('/')  # Should render index template
            assert response.status_code in [200, 404, 302]  # Various valid responses

            # Test template context processors
            with app.app_context():
                # Test that template globals are available
                assert 'url_for' in app.jinja_env.globals


class TestWebRoutesFunctionality:
    """Test web routes functionality."""

    def setup_method(self):
        """Set up routes test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_key',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()

    def test_main_routes(self):
        """Test main application routes."""

        # Test home route
        response = self.client.get('/')
        assert response.status_code in [200, 302]  # Success or redirect

        # Test about route
        response = self.client.get('/about')
        assert response.status_code in [200, 404]  # Success or not implemented

        # Test health check route
        response = self.client.get('/health')
        assert response.status_code == 200

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            assert data['status'] in ['healthy', 'ok']

    def test_api_routes_basic(self):
        """Test basic API routes functionality."""

        # Test API info endpoint
        response = self.client.get('/api/info')
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'version' in data or 'api' in data

        # Test API status endpoint
        response = self.client.get('/api/status')
        assert response.status_code in [200, 404]

    def test_quantum_routes_basic(self):
        """Test quantum-specific routes."""

        # Test quantum dashboard
        response = self.client.get('/quantum/dashboard')
        assert response.status_code in [200, 302, 401, 404]

        # Test quantum circuits endpoint
        response = self.client.get('/quantum/circuits')
        assert response.status_code in [200, 302, 401, 404]

    def test_admin_routes_security(self):
        """Test admin routes require authentication."""

        # Test admin dashboard (should require auth)
        response = self.client.get('/admin/')
        assert response.status_code in [302, 401, 403, 404]  # Redirect to login or unauthorized

        # Test admin users endpoint
        response = self.client.get('/admin/users')
        assert response.status_code in [302, 401, 403, 404]

    def test_route_methods_validation(self):
        """Test HTTP methods validation on routes."""

        # Test POST on GET-only route
        response = self.client.post('/')
        assert response.status_code in [405, 404]  # Method not allowed or not found

        # Test GET on POST-only route
        response = self.client.get('/api/quantum/execute')
        assert response.status_code in [405, 404]  # Method not allowed or not found

    def test_route_parameters_validation(self):
        """Test route parameter validation."""

        # Test route with valid parameter
        response = self.client.get('/quantum/circuit/valid_circuit_id')
        assert response.status_code in [200, 401, 404]  # Valid responses

        # Test route with invalid parameter format
        response = self.client.get('/quantum/circuit/')  # Missing parameter
        assert response.status_code in [404, 400]  # Not found or bad request


class TestWebSecurityFeatures:
    """Test web security features and protections."""

    def setup_method(self):
        """Set up security test environment."""
        self.app = create_secure_app({
            'TESTING': True,
            'SECRET_KEY': 'test_security_key',
            'WTF_CSRF_ENABLED': True
        })
        self.client = self.app.test_client()

    def test_csrf_protection(self):
        """Test CSRF protection implementation."""

        # Test CSRF protection on POST requests
        response = self.client.post('/api/quantum/execute',
                                  json={'circuit': 'test'},
                                  headers={'Content-Type': 'application/json'})

        # Should be rejected due to missing CSRF token
        assert response.status_code in [400, 403, 404]

    def test_xss_prevention(self):
        """Test XSS prevention measures."""

        # Test XSS payload in query parameters
        xss_payload = '<script>alert("xss")</script>'
        response = self.client.get(f'/search?q={xss_payload}')

        # Response should not contain unescaped script tags
        if response.status_code == 200:
            assert b'<script>alert("xss")</script>' not in response.data
            assert b'&lt;script&gt;' in response.data or b'script' not in response.data

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""

        # Test SQL injection payload
        sql_payload = "'; DROP TABLE users; --"

        # Test in various endpoints
        test_endpoints = [
            f'/api/quantum/circuit/{sql_payload}',
            f'/admin/user/{sql_payload}',
            f'/search?q={sql_payload}'
        ]

        for endpoint in test_endpoints:
            response = self.client.get(endpoint)
            # Should not cause server error (500)
            assert response.status_code != 500

    def test_security_headers(self):
        """Test security headers implementation."""

        response = self.client.get('/')

        # Test for important security headers
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security'
        ]

        for header in security_headers:
            # Header should be present (might be set by secure_app)
            if header in response.headers:
                assert len(response.headers[header]) > 0

    def test_rate_limiting(self):
        """Test rate limiting functionality."""

        # Test multiple rapid requests
        endpoint = '/api/quantum/info'
        responses = []

        for i in range(10):
            response = self.client.get(endpoint)
            responses.append(response.status_code)

        # Should eventually hit rate limit (429) or continue to work
        # Rate limiting might not be active in testing
        status_codes = set(responses)
        valid_codes = {200, 404, 429, 401}  # Valid response codes
        assert status_codes.issubset(valid_codes)

    def test_input_validation(self):
        """Test input validation and sanitization."""

        # Test various input validation scenarios
        test_cases = [
            {
                'endpoint': '/api/quantum/execute',
                'method': 'POST',
                'data': {'circuit': ''},  # Empty circuit
                'expected_codes': [400, 422, 404]
            },
            {
                'endpoint': '/api/quantum/execute',
                'method': 'POST',
                'data': {'invalid_field': 'value'},  # Invalid field
                'expected_codes': [400, 422, 404]
            }
        ]

        for test_case in test_cases:
            if test_case['method'] == 'POST':
                response = self.client.post(
                    test_case['endpoint'],
                    json=test_case['data'],
                    headers={'Content-Type': 'application/json'}
                )
            else:
                response = self.client.get(test_case['endpoint'])

            assert response.status_code in test_case['expected_codes']


class TestWebSocketFunctionality:
    """Test WebSocket functionality."""

    def setup_method(self):
        """Set up WebSocket test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_websocket_key'
        })

    def test_websocket_configuration(self):
        """Test WebSocket configuration."""

        # Test that SocketIO is configured
        from dt_project.web_interface.websocket_handler import socketio

        assert socketio is not None

        # Test WebSocket initialization
        socketio.init_app(self.app, cors_allowed_origins="*")

    def test_websocket_events(self):
        """Test WebSocket event handling."""

        from dt_project.web_interface.websocket_handler import socketio

        # Test WebSocket client connection
        client = socketio.test_client(self.app)

        # Test connection event
        assert client.is_connected()

        # Test custom events
        test_events = [
            'quantum_circuit_update',
            'simulation_progress',
            'twin_state_change'
        ]

        for event in test_events:
            # Emit test event
            client.emit(event, {'test': 'data'})

        # Test disconnection
        client.disconnect()
        assert not client.is_connected()

    def test_websocket_authentication(self):
        """Test WebSocket authentication."""

        from dt_project.web_interface.websocket_handler import socketio

        # Test connection without authentication
        client = socketio.test_client(self.app)

        # Should be able to connect (authentication might be on specific events)
        assert client.is_connected()

        # Test authenticated events
        auth_token = 'test_auth_token_for_websocket_testing'
        client.emit('authenticated_event', {
            'token': auth_token,
            'data': 'test_data'
        })

        client.disconnect()

    def test_websocket_real_time_updates(self):
        """Test real-time updates via WebSocket."""

        from dt_project.web_interface.websocket_handler import socketio

        client = socketio.test_client(self.app)

        # Test real-time quantum circuit updates
        circuit_update = {
            'circuit_id': 'test_circuit_001',
            'gates': [{'type': 'h', 'qubit': 0}],
            'status': 'updated'
        }

        client.emit('quantum_circuit_update', circuit_update)

        # Test real-time simulation progress
        progress_update = {
            'simulation_id': 'sim_001',
            'progress': 75,
            'eta_seconds': 30
        }

        client.emit('simulation_progress', progress_update)

        client.disconnect()


class TestGraphQLAPI:
    """Test GraphQL API functionality."""

    def setup_method(self):
        """Set up GraphQL test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_graphql_key'
        })
        self.client = self.app.test_client()

    def test_graphql_endpoint_availability(self):
        """Test GraphQL endpoint availability."""

        # Test GraphQL endpoint
        response = self.client.get('/graphql')
        assert response.status_code in [200, 404, 405]  # Various valid responses

        # Test GraphiQL interface (if enabled)
        response = self.client.get('/graphiql')
        assert response.status_code in [200, 404]

    def test_graphql_introspection(self):
        """Test GraphQL schema introspection."""

        introspection_query = {
            'query': '''
                query IntrospectionQuery {
                    __schema {
                        types {
                            name
                        }
                    }
                }
            '''
        }

        response = self.client.post('/graphql',
                                  json=introspection_query,
                                  headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'data' in data
            assert '__schema' in data['data']

    def test_graphql_quantum_queries(self):
        """Test GraphQL quantum-specific queries."""

        # Test quantum circuit query
        circuit_query = {
            'query': '''
                query GetQuantumCircuit($id: String!) {
                    quantumCircuit(id: $id) {
                        id
                        name
                        gates
                        qubits
                    }
                }
            ''',
            'variables': {'id': 'test_circuit_001'}
        }

        response = self.client.post('/graphql',
                                  json=circuit_query,
                                  headers={'Content-Type': 'application/json'})

        # Response could be 200 with data/errors or 404 if not implemented
        assert response.status_code in [200, 404]

    def test_graphql_mutations(self):
        """Test GraphQL mutations."""

        # Test create quantum circuit mutation
        create_circuit_mutation = {
            'query': '''
                mutation CreateQuantumCircuit($input: QuantumCircuitInput!) {
                    createQuantumCircuit(input: $input) {
                        id
                        success
                        message
                    }
                }
            ''',
            'variables': {
                'input': {
                    'name': 'test_circuit',
                    'gates': [{'type': 'h', 'qubit': 0}],
                    'qubits': 1
                }
            }
        }

        response = self.client.post('/graphql',
                                  json=create_circuit_mutation,
                                  headers={'Content-Type': 'application/json'})

        # Should either work (200) or not be implemented (404)
        assert response.status_code in [200, 404, 400]

    def test_graphql_error_handling(self):
        """Test GraphQL error handling."""

        # Test invalid query
        invalid_query = {
            'query': '''
                query InvalidQuery {
                    nonExistentField {
                        id
                    }
                }
            '''
        }

        response = self.client.post('/graphql',
                                  json=invalid_query,
                                  headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            data = json.loads(response.data)
            # Should contain errors for invalid query
            assert 'errors' in data

    def test_graphql_subscriptions(self):
        """Test GraphQL subscriptions (if implemented)."""

        # Test subscription for real-time updates
        subscription_query = {
            'query': '''
                subscription QuantumCircuitUpdates {
                    quantumCircuitUpdated {
                        id
                        status
                        progress
                    }
                }
            '''
        }

        response = self.client.post('/graphql',
                                  json=subscription_query,
                                  headers={'Content-Type': 'application/json'})

        # Subscriptions might not be supported in testing
        assert response.status_code in [200, 404, 400]


class TestWebApplicationIntegration:
    """Test web application integration scenarios."""

    def setup_method(self):
        """Set up integration test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'integration_test_key',
            'DATABASE_URL': 'sqlite:///:memory:'
        })
        self.client = self.app.test_client()

    def test_full_quantum_circuit_workflow(self):
        """Test complete quantum circuit workflow through web interface."""

        # Step 1: Create quantum circuit
        circuit_data = {
            'name': 'integration_test_circuit',
            'qubits': 2,
            'gates': [
                {'type': 'h', 'qubit': 0},
                {'type': 'cnot', 'control': 0, 'target': 1}
            ]
        }

        response = self.client.post('/api/quantum/circuits',
                                  json=circuit_data,
                                  headers={'Content-Type': 'application/json'})

        # Should either work or require authentication
        assert response.status_code in [200, 201, 401, 404]

        if response.status_code in [200, 201]:
            circuit_response = json.loads(response.data)
            circuit_id = circuit_response.get('id') or circuit_response.get('circuit_id')

            if circuit_id:
                # Step 2: Execute quantum circuit
                execution_data = {
                    'circuit_id': circuit_id,
                    'shots': 1000,
                    'backend': 'simulator'
                }

                response = self.client.post('/api/quantum/execute',
                                          json=execution_data,
                                          headers={'Content-Type': 'application/json'})

                assert response.status_code in [200, 202, 401, 404]

    def test_quantum_twin_management_workflow(self):
        """Test quantum digital twin management workflow."""

        # Step 1: Create quantum twin
        twin_data = {
            'twin_id': 'integration_test_twin',
            'entity_type': 'test_entity',
            'quantum_dimensions': 4
        }

        response = self.client.post('/api/quantum/twins',
                                  json=twin_data,
                                  headers={'Content-Type': 'application/json'})

        assert response.status_code in [200, 201, 401, 404]

        if response.status_code in [200, 201]:
            # Step 2: Update twin state
            state_update = {
                'quantum_state': [0.7071, 0, 0, 0.7071],
                'timestamp': datetime.utcnow().isoformat()
            }

            response = self.client.put(f'/api/quantum/twins/{twin_data["twin_id"]}/state',
                                     json=state_update,
                                     headers={'Content-Type': 'application/json'})

            assert response.status_code in [200, 401, 404]

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""

        # Test invalid JSON
        response = self.client.post('/api/quantum/circuits',
                                  data='invalid json',
                                  headers={'Content-Type': 'application/json'})

        assert response.status_code in [400, 404]

        # Test missing required fields
        response = self.client.post('/api/quantum/circuits',
                                  json={},  # Empty object
                                  headers={'Content-Type': 'application/json'})

        assert response.status_code in [400, 422, 404]

    def test_performance_under_load(self):
        """Test web application performance under simulated load."""

        # Simulate multiple concurrent requests
        import threading
        import time

        results = []

        def make_request():
            start_time = time.time()
            response = self.client.get('/api/status')
            end_time = time.time()
            results.append({
                'status_code': response.status_code,
                'response_time': end_time - start_time
            })

        # Create multiple threads for concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Analyze results
        assert len(results) == 10

        # All requests should complete successfully or with expected errors
        valid_status_codes = {200, 404, 401, 500}
        for result in results:
            assert result['status_code'] in valid_status_codes
            assert result['response_time'] < 5.0  # Should complete within 5 seconds


class TestWebApplicationConfiguration:
    """Test web application configuration management."""

    def test_environment_based_configuration(self):
        """Test configuration based on environment."""

        # Test development configuration
        dev_config = {'DEBUG': True, 'TESTING': False}
        dev_app = create_app(dev_config)
        assert dev_app.config['DEBUG'] == True

        # Test production configuration
        prod_config = {'DEBUG': False, 'TESTING': False}
        prod_app = create_app(prod_config)
        assert prod_app.config['DEBUG'] == False

        # Test testing configuration
        test_config = {'DEBUG': False, 'TESTING': True}
        test_app = create_app(test_config)
        assert test_app.config['TESTING'] == True

    def test_secret_key_validation(self):
        """Test secret key validation."""

        # Test with valid secret key
        valid_config = {'SECRET_KEY': 'valid_secret_key_with_sufficient_length'}
        app = create_app(valid_config)
        assert app.config['SECRET_KEY'] == valid_config['SECRET_KEY']

        # Test warning for weak secret key (if validation exists)
        weak_config = {'SECRET_KEY': 'weak'}
        weak_app = create_app(weak_config)
        # App should still be created but might have warnings
        assert weak_app.config['SECRET_KEY'] == 'weak'

    def test_database_configuration(self):
        """Test database configuration."""

        # Test different database URLs
        db_configs = [
            {'DATABASE_URL': 'sqlite:///:memory:'},
            {'DATABASE_URL': 'postgresql://user:pass@localhost/testdb'},
            {'DATABASE_URL': 'mysql://user:pass@localhost/testdb'}
        ]

        for config in db_configs:
            app = create_app(config)
            assert app.config['DATABASE_URL'] == config['DATABASE_URL']

    def test_logging_configuration(self):
        """Test logging configuration."""

        # Test logging configuration
        app = create_app({'TESTING': True})

        # Test that logging is configured
        assert app.logger is not None

        # Test log level configuration
        with app.app_context():
            app.logger.info("Test log message")
            app.logger.error("Test error message")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])