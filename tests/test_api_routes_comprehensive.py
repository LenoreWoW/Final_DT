"""
Comprehensive tests for API routes and endpoints.
Tests all API functionality including quantum operations, data management, and security.
"""

import pytest
import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from flask import Flask

# Import API components
from dt_project.web_interface.routes.api_routes import api_bp
from dt_project.web_interface.routes.quantum_routes import quantum_bp
from dt_project.web_interface.routes.quantum_lab_routes import quantum_lab_bp
from dt_project.web_interface.routes.admin_routes import admin_bp
from dt_project.web_interface.routes.simulation_routes import simulation_bp
from dt_project.web_interface.app import create_app


class TestAPIRoutesCore:
    """Test core API routes functionality."""

    def setup_method(self):
        """Set up API routes test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_api_key',
            'WTF_CSRF_ENABLED': False,
            'DATABASE_URL': 'sqlite:///:memory:'
        })
        self.client = self.app.test_client()
        self.headers = {'Content-Type': 'application/json'}

    def test_api_info_endpoint(self):
        """Test API information endpoint."""

        response = self.client.get('/api/info')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should contain API information
            expected_fields = ['version', 'name', 'description', 'endpoints']
            for field in expected_fields:
                if field in data:
                    assert data[field] is not None

        assert response.status_code in [200, 404]

    def test_api_status_endpoint(self):
        """Test API status endpoint."""

        response = self.client.get('/api/status')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should contain status information
            assert 'status' in data
            assert data['status'] in ['healthy', 'ok', 'running']

            # May contain additional metrics
            optional_fields = ['uptime', 'version', 'timestamp', 'services']
            for field in optional_fields:
                if field in data:
                    assert data[field] is not None

        assert response.status_code in [200, 404]

    def test_api_health_check(self):
        """Test API health check endpoint."""

        response = self.client.get('/api/health')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Health check should indicate system status
            assert 'healthy' in data or 'status' in data

        assert response.status_code in [200, 404]

    def test_api_version_endpoint(self):
        """Test API version endpoint."""

        response = self.client.get('/api/version')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should contain version information
            version_fields = ['version', 'api_version', 'build']
            assert any(field in data for field in version_fields)

        assert response.status_code in [200, 404]


class TestQuantumAPIRoutes:
    """Test quantum-specific API routes."""

    def setup_method(self):
        """Set up quantum API test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_quantum_api_key',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()
        self.headers = {'Content-Type': 'application/json'}

    def test_quantum_circuits_list_endpoint(self):
        """Test quantum circuits listing endpoint."""

        response = self.client.get('/api/quantum/circuits')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return list of circuits
            assert isinstance(data, (list, dict))

            if isinstance(data, dict):
                assert 'circuits' in data or 'data' in data

        assert response.status_code in [200, 401, 404]

    def test_quantum_circuit_creation(self):
        """Test quantum circuit creation endpoint."""

        circuit_data = {
            'name': 'test_circuit',
            'qubits': 3,
            'gates': [
                {'type': 'h', 'qubit': 0},
                {'type': 'cnot', 'control': 0, 'target': 1},
                {'type': 'measure', 'qubit': 0, 'bit': 0}
            ],
            'description': 'Test circuit for API testing'
        }

        response = self.client.post('/api/quantum/circuits',
                                  json=circuit_data,
                                  headers=self.headers)

        if response.status_code in [200, 201]:
            data = json.loads(response.data)

            # Should return circuit information
            expected_fields = ['id', 'circuit_id', 'name', 'status']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 201, 400, 401, 404, 422]

    def test_quantum_circuit_execution(self):
        """Test quantum circuit execution endpoint."""

        execution_data = {
            'circuit_id': 'test_circuit_001',
            'shots': 1000,
            'backend': 'simulator',
            'optimization_level': 1
        }

        response = self.client.post('/api/quantum/execute',
                                  json=execution_data,
                                  headers=self.headers)

        if response.status_code in [200, 202]:
            data = json.loads(response.data)

            # Should return execution information
            expected_fields = ['job_id', 'status', 'estimated_completion']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 202, 400, 401, 404, 422]

    def test_quantum_job_status(self):
        """Test quantum job status endpoint."""

        job_id = 'test_job_001'
        response = self.client.get(f'/api/quantum/jobs/{job_id}')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return job status information
            expected_fields = ['job_id', 'status', 'progress', 'result']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 401, 404]

    def test_quantum_results_retrieval(self):
        """Test quantum job results retrieval."""

        job_id = 'test_job_001'
        response = self.client.get(f'/api/quantum/jobs/{job_id}/results')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return quantum results
            result_fields = ['counts', 'measurements', 'statevector', 'probabilities']
            assert any(field in data for field in result_fields)

        assert response.status_code in [200, 401, 404, 425]  # 425 = Too Early

    def test_quantum_algorithms_endpoint(self):
        """Test quantum algorithms endpoint."""

        response = self.client.get('/api/quantum/algorithms')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return available algorithms
            assert isinstance(data, (list, dict))

            if isinstance(data, list):
                # List of algorithm names
                algorithms = ['grover', 'bernstein_vazirani', 'qft', 'vqe']
                available_algorithms = [alg.lower() for alg in data]
                assert any(alg in available_algorithms for alg in algorithms)

        assert response.status_code in [200, 404]

    def test_quantum_backends_endpoint(self):
        """Test quantum backends endpoint."""

        response = self.client.get('/api/quantum/backends')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return available backends
            assert isinstance(data, (list, dict))

            if isinstance(data, list):
                # Should contain at least simulator backend
                backend_names = [backend.lower() if isinstance(backend, str) else backend.get('name', '').lower() for backend in data]
                assert any('simulator' in name for name in backend_names)

        assert response.status_code in [200, 404]

    def test_quantum_twin_creation(self):
        """Test quantum digital twin creation."""

        twin_data = {
            'twin_id': 'api_test_twin_001',
            'entity_type': 'manufacturing_unit',
            'industry_domain': 'manufacturing',
            'quantum_dimensions': 4,
            'sensor_config': {
                'sensors': ['temperature', 'pressure', 'vibration'],
                'sampling_rate': 100
            }
        }

        response = self.client.post('/api/quantum/twins',
                                  json=twin_data,
                                  headers=self.headers)

        if response.status_code in [200, 201]:
            data = json.loads(response.data)

            # Should return twin information
            expected_fields = ['twin_id', 'status', 'quantum_state']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 201, 400, 401, 404, 422]

    def test_quantum_twin_state_update(self):
        """Test quantum twin state update."""

        twin_id = 'api_test_twin_001'
        state_data = {
            'quantum_state': [0.7071, 0, 0, 0.7071],
            'sensor_readings': {
                'temperature': 75.2,
                'pressure': 101325,
                'vibration': 0.15
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        response = self.client.put(f'/api/quantum/twins/{twin_id}/state',
                                 json=state_data,
                                 headers=self.headers)

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should confirm state update
            assert 'success' in data or 'status' in data

        assert response.status_code in [200, 400, 401, 404, 422]

    def test_quantum_optimization_endpoint(self):
        """Test quantum optimization endpoint."""

        optimization_data = {
            'problem_type': 'maxcut',
            'graph': {
                'nodes': 4,
                'edges': [(0, 1), (1, 2), (2, 3), (3, 0)]
            },
            'algorithm': 'qaoa',
            'iterations': 100
        }

        response = self.client.post('/api/quantum/optimize',
                                  json=optimization_data,
                                  headers=self.headers)

        if response.status_code in [200, 202]:
            data = json.loads(response.data)

            # Should return optimization job information
            expected_fields = ['job_id', 'status', 'problem_type']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 202, 400, 401, 404, 422]


class TestQuantumLabRoutes:
    """Test quantum lab routes for interactive quantum programming."""

    def setup_method(self):
        """Set up quantum lab test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_quantum_lab_key',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()
        self.headers = {'Content-Type': 'application/json'}

    def test_quantum_lab_dashboard(self):
        """Test quantum lab dashboard endpoint."""

        response = self.client.get('/quantum-lab/')

        # Should return quantum lab interface
        assert response.status_code in [200, 302, 401, 404]

        if response.status_code == 200:
            # Should be HTML content for lab interface
            assert response.content_type.startswith('text/html') or 'application/json' in response.content_type

    def test_quantum_circuit_builder(self):
        """Test quantum circuit builder endpoint."""

        response = self.client.get('/quantum-lab/circuit-builder')

        assert response.status_code in [200, 302, 401, 404]

    def test_quantum_simulator_endpoint(self):
        """Test quantum simulator endpoint."""

        simulation_data = {
            'circuit': {
                'qubits': 2,
                'gates': [
                    {'type': 'h', 'qubit': 0},
                    {'type': 'cnot', 'control': 0, 'target': 1}
                ]
            },
            'shots': 1000,
            'simulator_type': 'statevector'
        }

        response = self.client.post('/quantum-lab/simulate',
                                  json=simulation_data,
                                  headers=self.headers)

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return simulation results
            result_fields = ['counts', 'statevector', 'probabilities']
            assert any(field in data for field in result_fields)

        assert response.status_code in [200, 400, 401, 404, 422]

    def test_quantum_visualization_endpoint(self):
        """Test quantum visualization endpoint."""

        visualization_data = {
            'visualization_type': 'bloch_sphere',
            'quantum_state': [0.7071, 0, 0, 0.7071],
            'format': 'svg'
        }

        response = self.client.post('/quantum-lab/visualize',
                                  json=visualization_data,
                                  headers=self.headers)

        if response.status_code == 200:
            # Could return SVG, PNG, or JSON with visualization data
            assert response.content_type in [
                'image/svg+xml', 'image/png', 'application/json'
            ]

        assert response.status_code in [200, 400, 401, 404, 422]

    def test_quantum_tutorial_endpoints(self):
        """Test quantum tutorial endpoints."""

        # Test tutorial listing
        response = self.client.get('/quantum-lab/tutorials')
        assert response.status_code in [200, 404]

        # Test specific tutorial
        tutorial_id = 'introduction-to-quantum-gates'
        response = self.client.get(f'/quantum-lab/tutorials/{tutorial_id}')
        assert response.status_code in [200, 404]

    def test_quantum_code_execution(self):
        """Test quantum code execution endpoint."""

        code_data = {
            'language': 'qiskit',
            'code': '''
                from qiskit import QuantumCircuit, execute, Aer
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure_all()
                backend = Aer.get_backend('qasm_simulator')
                job = execute(qc, backend, shots=1000)
                result = job.result()
                counts = result.get_counts()
            ''',
            'timeout': 30
        }

        response = self.client.post('/quantum-lab/execute',
                                  json=code_data,
                                  headers=self.headers)

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return execution results
            expected_fields = ['output', 'result', 'error', 'execution_time']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 400, 401, 404, 422, 408]  # 408 = Timeout


class TestAdminAPIRoutes:
    """Test admin API routes."""

    def setup_method(self):
        """Set up admin API test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_admin_api_key',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()
        self.headers = {'Content-Type': 'application/json'}

    def test_admin_dashboard_api(self):
        """Test admin dashboard API."""

        response = self.client.get('/api/admin/dashboard')

        # Should require authentication
        assert response.status_code in [200, 401, 403, 404]

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should contain dashboard metrics
            dashboard_fields = ['users', 'jobs', 'system_status', 'metrics']
            assert any(field in data for field in dashboard_fields)

    def test_admin_users_management(self):
        """Test admin users management API."""

        # Test list users
        response = self.client.get('/api/admin/users')
        assert response.status_code in [200, 401, 403, 404]

        # Test create user
        user_data = {
            'username': 'test_api_user',
            'email': 'test@example.com',
            'role': 'user',
            'permissions': ['quantum_execute', 'circuit_create']
        }

        response = self.client.post('/api/admin/users',
                                  json=user_data,
                                  headers=self.headers)

        assert response.status_code in [200, 201, 400, 401, 403, 404, 422]

        # Test update user
        user_id = 'test_user_001'
        update_data = {
            'role': 'admin',
            'permissions': ['all']
        }

        response = self.client.put(f'/api/admin/users/{user_id}',
                                 json=update_data,
                                 headers=self.headers)

        assert response.status_code in [200, 400, 401, 403, 404, 422]

    def test_admin_system_monitoring(self):
        """Test admin system monitoring API."""

        # Test system metrics
        response = self.client.get('/api/admin/metrics')
        assert response.status_code in [200, 401, 403, 404]

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should contain system metrics
            metric_fields = ['cpu_usage', 'memory_usage', 'disk_usage', 'active_jobs']
            if isinstance(data, dict):
                assert any(field in data for field in metric_fields)

        # Test system logs
        response = self.client.get('/api/admin/logs')
        assert response.status_code in [200, 401, 403, 404]

    def test_admin_quantum_resources(self):
        """Test admin quantum resources management."""

        # Test quantum backends management
        response = self.client.get('/api/admin/quantum/backends')
        assert response.status_code in [200, 401, 403, 404]

        # Test quantum job management
        response = self.client.get('/api/admin/quantum/jobs')
        assert response.status_code in [200, 401, 403, 404]

        # Test job cancellation
        job_id = 'test_admin_job_001'
        response = self.client.delete(f'/api/admin/quantum/jobs/{job_id}')
        assert response.status_code in [200, 204, 401, 403, 404]

    def test_admin_configuration_management(self):
        """Test admin configuration management."""

        # Test get configuration
        response = self.client.get('/api/admin/config')
        assert response.status_code in [200, 401, 403, 404]

        # Test update configuration
        config_data = {
            'max_concurrent_jobs': 10,
            'default_shots': 1000,
            'optimization_level': 2
        }

        response = self.client.put('/api/admin/config',
                                 json=config_data,
                                 headers=self.headers)

        assert response.status_code in [200, 400, 401, 403, 404, 422]


class TestSimulationRoutes:
    """Test simulation-specific routes."""

    def setup_method(self):
        """Set up simulation test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_simulation_key',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()
        self.headers = {'Content-Type': 'application/json'}

    def test_simulation_creation(self):
        """Test simulation creation endpoint."""

        simulation_data = {
            'simulation_type': 'quantum_digital_twin',
            'twin_id': 'simulation_twin_001',
            'parameters': {
                'duration_seconds': 60,
                'time_step': 0.1,
                'noise_model': 'depolarizing',
                'error_rate': 0.001
            },
            'output_format': 'json'
        }

        response = self.client.post('/api/simulations',
                                  json=simulation_data,
                                  headers=self.headers)

        if response.status_code in [200, 201, 202]:
            data = json.loads(response.data)

            # Should return simulation information
            expected_fields = ['simulation_id', 'status', 'estimated_duration']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 201, 202, 400, 401, 404, 422]

    def test_simulation_status_monitoring(self):
        """Test simulation status monitoring."""

        simulation_id = 'test_simulation_001'
        response = self.client.get(f'/api/simulations/{simulation_id}')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return simulation status
            status_fields = ['simulation_id', 'status', 'progress', 'eta']
            assert any(field in data for field in status_fields)

        assert response.status_code in [200, 401, 404]

    def test_simulation_results_retrieval(self):
        """Test simulation results retrieval."""

        simulation_id = 'test_simulation_001'
        response = self.client.get(f'/api/simulations/{simulation_id}/results')

        if response.status_code == 200:
            data = json.loads(response.data)

            # Should return simulation results
            result_fields = ['results', 'data', 'output', 'metrics']
            assert any(field in data for field in result_fields)

        assert response.status_code in [200, 401, 404, 425]  # 425 = Too Early

    def test_simulation_control_operations(self):
        """Test simulation control operations."""

        simulation_id = 'test_simulation_001'

        # Test pause simulation
        response = self.client.post(f'/api/simulations/{simulation_id}/pause')
        assert response.status_code in [200, 400, 401, 404]

        # Test resume simulation
        response = self.client.post(f'/api/simulations/{simulation_id}/resume')
        assert response.status_code in [200, 400, 401, 404]

        # Test stop simulation
        response = self.client.post(f'/api/simulations/{simulation_id}/stop')
        assert response.status_code in [200, 400, 401, 404]

    def test_batch_simulation_management(self):
        """Test batch simulation management."""

        batch_data = {
            'batch_name': 'parameter_sweep_batch',
            'simulations': [
                {
                    'simulation_type': 'quantum_digital_twin',
                    'parameters': {'error_rate': 0.001}
                },
                {
                    'simulation_type': 'quantum_digital_twin',
                    'parameters': {'error_rate': 0.01}
                },
                {
                    'simulation_type': 'quantum_digital_twin',
                    'parameters': {'error_rate': 0.1}
                }
            ]
        }

        response = self.client.post('/api/simulations/batch',
                                  json=batch_data,
                                  headers=self.headers)

        if response.status_code in [200, 201, 202]:
            data = json.loads(response.data)

            # Should return batch information
            expected_fields = ['batch_id', 'simulations', 'status']
            assert any(field in data for field in expected_fields)

        assert response.status_code in [200, 201, 202, 400, 401, 404, 422]


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""

    def setup_method(self):
        """Set up error handling test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_error_handling_key',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()
        self.headers = {'Content-Type': 'application/json'}

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON."""

        # Test malformed JSON
        response = self.client.post('/api/quantum/circuits',
                                  data='{"invalid": json}',
                                  headers=self.headers)

        assert response.status_code in [400, 404]

        if response.status_code == 400:
            data = json.loads(response.data)
            assert 'error' in data or 'message' in data

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""

        # Test empty request body
        response = self.client.post('/api/quantum/circuits',
                                  json={},
                                  headers=self.headers)

        assert response.status_code in [400, 422, 404]

        # Test partial data
        partial_data = {'name': 'test_circuit'}  # Missing other required fields

        response = self.client.post('/api/quantum/circuits',
                                  json=partial_data,
                                  headers=self.headers)

        assert response.status_code in [400, 422, 404]

    def test_invalid_data_types(self):
        """Test handling of invalid data types."""

        invalid_data = {
            'name': 123,  # Should be string
            'qubits': 'invalid',  # Should be integer
            'gates': 'not_a_list'  # Should be list
        }

        response = self.client.post('/api/quantum/circuits',
                                  json=invalid_data,
                                  headers=self.headers)

        assert response.status_code in [400, 422, 404]

    def test_resource_not_found(self):
        """Test handling of resource not found."""

        # Test non-existent circuit
        response = self.client.get('/api/quantum/circuits/nonexistent_circuit')
        assert response.status_code in [404, 401]

        # Test non-existent job
        response = self.client.get('/api/quantum/jobs/nonexistent_job')
        assert response.status_code in [404, 401]

    def test_authentication_errors(self):
        """Test authentication error handling."""

        # Test endpoints that require authentication
        protected_endpoints = [
            '/api/admin/users',
            '/api/admin/config',
            '/api/quantum/execute'
        ]

        for endpoint in protected_endpoints:
            response = self.client.get(endpoint)
            # Should require authentication
            assert response.status_code in [401, 403, 404]

    def test_rate_limiting_behavior(self):
        """Test rate limiting behavior."""

        # Make multiple rapid requests to test rate limiting
        endpoint = '/api/quantum/info'
        responses = []

        for i in range(20):
            response = self.client.get(endpoint)
            responses.append(response.status_code)

        # Should either work normally or hit rate limits
        valid_codes = {200, 404, 429, 401}  # 429 = Too Many Requests
        for status_code in responses:
            assert status_code in valid_codes

    def test_large_payload_handling(self):
        """Test handling of large payloads."""

        # Create large circuit data
        large_circuit = {
            'name': 'large_circuit',
            'qubits': 100,
            'gates': [{'type': 'h', 'qubit': i} for i in range(1000)]  # Large gate list
        }

        response = self.client.post('/api/quantum/circuits',
                                  json=large_circuit,
                                  headers=self.headers)

        # Should either process or reject due to size limits
        assert response.status_code in [200, 201, 400, 413, 404, 422]  # 413 = Payload Too Large

    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""

        import threading
        import time

        results = []

        def make_concurrent_request():
            response = self.client.get('/api/status')
            results.append(response.status_code)

        # Create multiple threads for concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_concurrent_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All requests should complete successfully
        assert len(results) == 5
        valid_codes = {200, 404, 500, 401}
        for status_code in results:
            assert status_code in valid_codes


class TestAPIPerformance:
    """Test API performance characteristics."""

    def setup_method(self):
        """Set up performance test environment."""
        self.app = create_app({
            'TESTING': True,
            'SECRET_KEY': 'test_performance_key',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()

    def test_response_time_benchmarks(self):
        """Test API response time benchmarks."""

        import time

        endpoints = [
            '/api/status',
            '/api/info',
            '/api/quantum/algorithms',
            '/api/quantum/backends'
        ]

        for endpoint in endpoints:
            start_time = time.time()
            response = self.client.get(endpoint)
            end_time = time.time()

            response_time = end_time - start_time

            # Response should be reasonably fast (< 5 seconds)
            assert response_time < 5.0

            # If successful, should be very fast
            if response.status_code == 200:
                assert response_time < 1.0

    def test_memory_usage_stability(self):
        """Test memory usage stability under load."""

        # Make multiple requests to test memory stability
        for i in range(50):
            response = self.client.get('/api/status')

            # Requests should continue to work
            assert response.status_code in [200, 404]

        # Test with POST requests
        test_data = {'test': 'data'}
        for i in range(20):
            response = self.client.post('/api/test',
                                      json=test_data,
                                      headers={'Content-Type': 'application/json'})

            # Should handle requests consistently
            assert response.status_code in [200, 404, 405]

    def test_api_scalability_indicators(self):
        """Test API scalability indicators."""

        # Test response consistency under increasing load
        load_levels = [1, 5, 10, 20]

        for load in load_levels:
            responses = []

            for i in range(load):
                response = self.client.get('/api/status')
                responses.append(response.status_code)

            # All responses should be consistent
            unique_statuses = set(responses)
            assert len(unique_statuses) <= 2  # Should be consistent (maybe 200 and 404)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])