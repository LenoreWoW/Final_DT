"""
Comprehensive tests for real quantum hardware integration.
Tests actual connections to quantum cloud providers and hardware management.
"""

import pytest
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import asdict

# Import quantum hardware integration components
from dt_project.core.real_quantum_hardware_integration import (
    QuantumProvider, QuantumHardwareType, QuantumDeviceSpecs,
    QuantumJob, JobStatus, QuantumCredentials, QuantumHardwareManager,
    IBMQuantumProvider, AmazonBraketProvider, GoogleQuantumProvider,
    AzureQuantumProvider, RigettiProvider, IonQProvider,
    QuantumResourceManager, HardwareOptimization, JobScheduler,
    QuantumCircuitTranspiler, ErrorMitigation, CalibrationManager,
    HardwareError, ProviderConnectionError, JobExecutionError,
    CredentialError, DeviceUnavailableError
)


class TestQuantumDeviceSpecs:
    """Test quantum device specifications and validation."""

    def test_device_specs_creation(self):
        """Test creation of quantum device specifications."""

        # Test IBM Quantum device specs
        ibm_specs = QuantumDeviceSpecs(
            provider=QuantumProvider.IBM,
            device_name='ibm_brisbane',
            n_qubits=127,
            hardware_type=QuantumHardwareType.GATE_BASED,
            connectivity='heavy_hex',
            gate_fidelity=0.999,
            coherence_time_us=100.0,
            gate_time_ns=160.0,
            max_shots=100000,
            queue_time_estimate_minutes=30
        )

        assert ibm_specs.provider == QuantumProvider.IBM
        assert ibm_specs.device_name == 'ibm_brisbane'
        assert ibm_specs.n_qubits == 127
        assert ibm_specs.hardware_type == QuantumHardwareType.GATE_BASED
        assert ibm_specs.gate_fidelity == 0.999

        # Test Google Quantum device specs
        google_specs = QuantumDeviceSpecs(
            provider=QuantumProvider.GOOGLE,
            device_name='weber',
            n_qubits=70,
            hardware_type=QuantumHardwareType.GATE_BASED,
            connectivity='grid',
            gate_fidelity=0.997,
            coherence_time_us=80.0,
            gate_time_ns=25.0,
            max_shots=1000000,
            queue_time_estimate_minutes=15
        )

        assert google_specs.provider == QuantumProvider.GOOGLE
        assert google_specs.n_qubits == 70
        assert google_specs.connectivity == 'grid'

    def test_device_specs_validation(self):
        """Test validation of device specifications."""

        # Test invalid specifications
        with pytest.raises(ValueError):
            QuantumDeviceSpecs(
                provider=QuantumProvider.IBM,
                device_name='test_device',
                n_qubits=-5,  # Invalid: negative qubits
                hardware_type=QuantumHardwareType.GATE_BASED,
                connectivity='linear',
                gate_fidelity=0.99,
                coherence_time_us=50.0,
                gate_time_ns=100.0
            )

        # Test invalid fidelity
        with pytest.raises(ValueError):
            QuantumDeviceSpecs(
                provider=QuantumProvider.IBM,
                device_name='test_device',
                n_qubits=5,
                hardware_type=QuantumHardwareType.GATE_BASED,
                connectivity='linear',
                gate_fidelity=1.5,  # Invalid: > 1.0
                coherence_time_us=50.0,
                gate_time_ns=100.0
            )

    def test_device_performance_metrics(self):
        """Test device performance metric calculations."""

        device_specs = QuantumDeviceSpecs(
            provider=QuantumProvider.IBM,
            device_name='test_device',
            n_qubits=20,
            hardware_type=QuantumHardwareType.GATE_BASED,
            connectivity='linear',
            gate_fidelity=0.995,
            coherence_time_us=75.0,
            gate_time_ns=200.0
        )

        # Test performance calculations
        performance_score = device_specs.calculate_performance_score()
        assert 0 <= performance_score <= 1

        quantum_volume = device_specs.calculate_quantum_volume()
        assert quantum_volume > 0

        error_rate = device_specs.calculate_error_rate()
        assert 0 <= error_rate <= 1


class TestQuantumCredentials:
    """Test quantum provider credential management."""

    def test_credential_creation_and_validation(self):
        """Test creation and validation of quantum credentials."""

        # Test IBM credentials
        ibm_credentials = QuantumCredentials(
            provider=QuantumProvider.IBM,
            api_token='fake_ibm_token_12345',
            instance='h/g/p',
            hub='ibm-q',
            group='open',
            project='main'
        )

        assert ibm_credentials.provider == QuantumProvider.IBM
        assert ibm_credentials.is_valid() == True

        # Test Amazon Braket credentials
        braket_credentials = QuantumCredentials(
            provider=QuantumProvider.AMAZON,
            aws_access_key_id='AKIAIOSFODNN7EXAMPLE',
            aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            aws_region='us-east-1',
            s3_bucket='amazon-braket-test-bucket'
        )

        assert braket_credentials.provider == QuantumProvider.AMAZON
        assert braket_credentials.is_valid() == True

    def test_credential_encryption(self):
        """Test credential encryption and decryption."""

        credentials = QuantumCredentials(
            provider=QuantumProvider.GOOGLE,
            service_account_key='{"type": "service_account", "project_id": "test"}',
            project_id='quantum-test-project'
        )

        # Test encryption
        encrypted_creds = credentials.encrypt()
        assert encrypted_creds != credentials.service_account_key

        # Test decryption
        decrypted_creds = QuantumCredentials.decrypt(encrypted_creds)
        assert decrypted_creds.service_account_key == credentials.service_account_key

    def test_credential_security_validation(self):
        """Test credential security validation."""

        # Test weak credential detection
        weak_credentials = QuantumCredentials(
            provider=QuantumProvider.IBM,
            api_token='weak123',  # Too short
            instance='h/g/p'
        )

        validation_result = weak_credentials.validate_security()
        assert validation_result.is_secure == False
        assert 'token_too_short' in validation_result.security_issues

        # Test strong credentials
        strong_credentials = QuantumCredentials(
            provider=QuantumProvider.IBM,
            api_token='very_long_secure_token_with_random_characters_12345abcdef',
            instance='h/g/p'
        )

        validation_result = strong_credentials.validate_security()
        assert validation_result.is_secure == True


class TestQuantumJob:
    """Test quantum job management and execution."""

    def test_quantum_job_creation(self):
        """Test quantum job creation and initialization."""

        job_config = {
            'job_id': 'quantum_job_001',
            'circuit_name': 'grover_search_circuit',
            'provider': QuantumProvider.IBM,
            'device_name': 'ibm_brisbane',
            'shots': 1000,
            'optimization_level': 3
        }

        quantum_job = QuantumJob(job_config)

        assert quantum_job.job_id == 'quantum_job_001'
        assert quantum_job.provider == QuantumProvider.IBM
        assert quantum_job.status == JobStatus.CREATED
        assert quantum_job.shots == 1000

    @pytest.mark.asyncio
    async def test_job_lifecycle_management(self):
        """Test complete job lifecycle management."""

        job_config = {
            'job_id': 'lifecycle_test_job',
            'circuit_name': 'test_circuit',
            'provider': QuantumProvider.IBM,
            'device_name': 'ibm_kyoto',
            'shots': 500
        }

        quantum_job = QuantumJob(job_config)

        # Test job submission
        submission_result = await quantum_job.submit()
        assert submission_result.success == True
        assert quantum_job.status == JobStatus.QUEUED

        # Test job monitoring
        with patch('dt_project.core.real_quantum_hardware_integration.time.sleep'):
            monitoring_result = await quantum_job.monitor_execution()
            assert monitoring_result.final_status in [JobStatus.COMPLETED, JobStatus.FAILED]

        # Test result retrieval
        if quantum_job.status == JobStatus.COMPLETED:
            results = await quantum_job.get_results()
            assert results is not None
            assert 'counts' in results or 'measurements' in results

    def test_job_error_handling(self):
        """Test job error handling and recovery."""

        job_config = {
            'job_id': 'error_test_job',
            'circuit_name': 'invalid_circuit',
            'provider': QuantumProvider.IBM,
            'device_name': 'nonexistent_device',
            'shots': 1000
        }

        quantum_job = QuantumJob(job_config)

        # Test error status handling
        quantum_job.set_error_status('Device not available')
        assert quantum_job.status == JobStatus.FAILED
        assert quantum_job.error_message == 'Device not available'

        # Test retry mechanism
        retry_result = quantum_job.retry_job()
        assert retry_result.retry_attempted == True
        assert quantum_job.retry_count == 1

    def test_job_priority_and_scheduling(self):
        """Test job priority and scheduling features."""

        high_priority_job = QuantumJob({
            'job_id': 'high_priority_job',
            'priority': 'high',
            'max_wait_time_minutes': 60
        })

        low_priority_job = QuantumJob({
            'job_id': 'low_priority_job',
            'priority': 'low',
            'max_wait_time_minutes': 1440  # 24 hours
        })

        # Test priority comparison
        assert high_priority_job.priority_score() > low_priority_job.priority_score()

        # Test deadline calculation
        high_deadline = high_priority_job.calculate_deadline()
        low_deadline = low_priority_job.calculate_deadline()

        assert high_deadline < low_deadline  # High priority has earlier deadline


class TestQuantumHardwareManager:
    """Test the main quantum hardware management system."""

    def setup_method(self):
        """Set up hardware manager test environment."""
        self.manager_config = {
            'supported_providers': ['ibm', 'amazon', 'google', 'azure'],
            'credential_storage': 'encrypted_file',
            'job_timeout_minutes': 60,
            'max_concurrent_jobs': 10,
            'auto_failover': True
        }

    @pytest.mark.asyncio
    async def test_hardware_manager_initialization(self):
        """Test hardware manager initialization."""

        with patch.multiple(
            'dt_project.core.real_quantum_hardware_integration',
            IBM_AVAILABLE=True,
            BRAKET_AVAILABLE=True,
            GOOGLE_AVAILABLE=True,
            AZURE_AVAILABLE=True
        ):
            manager = QuantumHardwareManager(self.manager_config)
            await manager.initialize()

            assert manager.is_initialized == True
            assert len(manager.provider_connections) > 0
            assert manager.job_scheduler is not None

    @pytest.mark.asyncio
    async def test_device_discovery_and_cataloging(self):
        """Test automatic device discovery across providers."""

        with patch.multiple(
            'dt_project.core.real_quantum_hardware_integration',
            IBMQuantumProvider=Mock,
            AmazonBraketProvider=Mock,
            GoogleQuantumProvider=Mock
        ) as mocks:

            # Mock provider device lists
            ibm_mock = Mock()
            ibm_mock.get_available_devices.return_value = [
                QuantumDeviceSpecs(
                    provider=QuantumProvider.IBM,
                    device_name='ibm_brisbane',
                    n_qubits=127,
                    hardware_type=QuantumHardwareType.GATE_BASED,
                    connectivity='heavy_hex',
                    gate_fidelity=0.999,
                    coherence_time_us=100.0,
                    gate_time_ns=160.0
                )
            ]

            braket_mock = Mock()
            braket_mock.get_available_devices.return_value = [
                QuantumDeviceSpecs(
                    provider=QuantumProvider.AMAZON,
                    device_name='ionq_aria',
                    n_qubits=25,
                    hardware_type=QuantumHardwareType.TRAPPED_ION,
                    connectivity='full',
                    gate_fidelity=0.995,
                    coherence_time_us=10000.0,
                    gate_time_ns=10000.0
                )
            ]

            mocks['IBMQuantumProvider'].return_value = ibm_mock
            mocks['AmazonBraketProvider'].return_value = braket_mock

            manager = QuantumHardwareManager(self.manager_config)
            await manager.initialize()

            # Test device discovery
            discovery_result = await manager.discover_available_devices()

            assert discovery_result.discovery_successful == True
            assert len(discovery_result.discovered_devices) >= 2
            assert any(d.provider == QuantumProvider.IBM for d in discovery_result.discovered_devices)
            assert any(d.provider == QuantumProvider.AMAZON for d in discovery_result.discovered_devices)

    @pytest.mark.asyncio
    async def test_optimal_device_selection(self):
        """Test optimal device selection for quantum circuits."""

        manager = QuantumHardwareManager(self.manager_config)

        # Mock available devices with different characteristics
        mock_devices = [
            QuantumDeviceSpecs(
                provider=QuantumProvider.IBM,
                device_name='ibm_small',
                n_qubits=5,
                hardware_type=QuantumHardwareType.GATE_BASED,
                connectivity='linear',
                gate_fidelity=0.995,
                coherence_time_us=50.0,
                gate_time_ns=200.0,
                queue_time_estimate_minutes=5
            ),
            QuantumDeviceSpecs(
                provider=QuantumProvider.GOOGLE,
                device_name='google_large',
                n_qubits=70,
                hardware_type=QuantumHardwareType.GATE_BASED,
                connectivity='grid',
                gate_fidelity=0.997,
                coherence_time_us=80.0,
                gate_time_ns=25.0,
                queue_time_estimate_minutes=45
            )
        ]

        manager.available_devices = mock_devices

        # Test selection for small circuit
        small_circuit_requirements = {
            'required_qubits': 4,
            'required_connectivity': 'any',
            'max_queue_time_minutes': 10,
            'min_fidelity': 0.99
        }

        selection_result = await manager.select_optimal_device(small_circuit_requirements)
        assert selection_result.selected_device.device_name == 'ibm_small'

        # Test selection for large circuit
        large_circuit_requirements = {
            'required_qubits': 50,
            'required_connectivity': 'grid',
            'max_queue_time_minutes': 60,
            'min_fidelity': 0.995
        }

        selection_result = await manager.select_optimal_device(large_circuit_requirements)
        assert selection_result.selected_device.device_name == 'google_large'

    @pytest.mark.asyncio
    async def test_circuit_transpilation_optimization(self):
        """Test quantum circuit transpilation for different hardware."""

        manager = QuantumHardwareManager(self.manager_config)

        # Mock quantum circuit
        mock_circuit = {
            'qubits': 4,
            'gates': [
                {'type': 'h', 'qubit': 0},
                {'type': 'cnot', 'control': 0, 'target': 1},
                {'type': 'cnot', 'control': 1, 'target': 2},
                {'type': 'measure', 'qubit': 0, 'bit': 0}
            ]
        }

        # Mock target device
        target_device = QuantumDeviceSpecs(
            provider=QuantumProvider.IBM,
            device_name='ibm_test',
            n_qubits=5,
            hardware_type=QuantumHardwareType.GATE_BASED,
            connectivity='linear',
            gate_fidelity=0.99,
            coherence_time_us=75.0,
            gate_time_ns=150.0
        )

        transpiler = QuantumCircuitTranspiler(manager)

        # Test transpilation
        transpilation_result = await transpiler.transpile_circuit(
            mock_circuit, target_device
        )

        assert transpilation_result.transpilation_successful == True
        assert transpilation_result.optimized_circuit is not None
        assert transpilation_result.gate_count_reduction >= 0
        assert transpilation_result.depth_reduction >= 0

    @pytest.mark.asyncio
    async def test_error_mitigation_strategies(self):
        """Test quantum error mitigation implementation."""

        manager = QuantumHardwareManager(self.manager_config)

        error_mitigation = ErrorMitigation(manager)

        # Test zero-noise extrapolation
        noisy_results = [
            {'0': 480, '1': 520},  # No mitigation
            {'0': 460, '1': 540},  # Light mitigation
            {'0': 440, '1': 560}   # Heavy mitigation
        ]

        zne_result = await error_mitigation.zero_noise_extrapolation(noisy_results)

        assert zne_result.mitigation_successful == True
        assert zne_result.error_reduction_factor > 1.0
        assert '0' in zne_result.mitigated_counts
        assert '1' in zne_result.mitigated_counts

        # Test readout error mitigation
        calibration_data = {
            'confusion_matrix': [[0.95, 0.05], [0.03, 0.97]],
            'measurement_fidelity': 0.96
        }

        readout_correction = await error_mitigation.readout_error_mitigation(
            noisy_results[0], calibration_data
        )

        assert readout_correction.correction_successful == True
        assert readout_correction.fidelity_improvement > 0

    @pytest.mark.asyncio
    async def test_job_scheduling_and_load_balancing(self):
        """Test job scheduling and load balancing across providers."""

        manager = QuantumHardwareManager(self.manager_config)

        scheduler = JobScheduler(manager)

        # Create multiple jobs
        jobs = []
        for i in range(5):
            job_config = {
                'job_id': f'scheduled_job_{i:03d}',
                'circuit_name': f'test_circuit_{i}',
                'shots': 1000,
                'priority': 'normal' if i < 3 else 'high'
            }
            jobs.append(QuantumJob(job_config))

        # Test job scheduling
        scheduling_result = await scheduler.schedule_jobs(jobs)

        assert scheduling_result.scheduling_successful == True
        assert len(scheduling_result.scheduled_jobs) == 5

        # Verify high-priority jobs scheduled first
        high_priority_jobs = [j for j in scheduling_result.scheduled_jobs if j.priority == 'high']
        normal_priority_jobs = [j for j in scheduling_result.scheduled_jobs if j.priority == 'normal']

        if high_priority_jobs and normal_priority_jobs:
            assert min(j.scheduled_time for j in high_priority_jobs) <= \
                   min(j.scheduled_time for j in normal_priority_jobs)

    @pytest.mark.asyncio
    async def test_hardware_calibration_monitoring(self):
        """Test hardware calibration and performance monitoring."""

        manager = QuantumHardwareManager(self.manager_config)

        calibration_manager = CalibrationManager(manager)

        # Test calibration data collection
        device_name = 'test_device'
        calibration_result = await calibration_manager.collect_calibration_data(device_name)

        assert calibration_result.collection_successful == True
        assert calibration_result.gate_fidelities is not None
        assert calibration_result.readout_fidelities is not None
        assert calibration_result.coherence_times is not None

        # Test performance drift detection
        historical_data = [
            {'timestamp': datetime.utcnow() - timedelta(days=7), 'fidelity': 0.999},
            {'timestamp': datetime.utcnow() - timedelta(days=3), 'fidelity': 0.997},
            {'timestamp': datetime.utcnow() - timedelta(days=1), 'fidelity': 0.995}
        ]

        drift_analysis = await calibration_manager.analyze_performance_drift(
            device_name, historical_data
        )

        assert drift_analysis.drift_detected == True
        assert drift_analysis.drift_rate < 0  # Decreasing performance

    @pytest.mark.asyncio
    async def test_provider_failover_mechanisms(self):
        """Test automatic failover between quantum providers."""

        manager = QuantumHardwareManager(self.manager_config)

        # Simulate provider failure
        job_config = {
            'job_id': 'failover_test_job',
            'preferred_provider': QuantumProvider.IBM,
            'fallback_providers': [QuantumProvider.GOOGLE, QuantumProvider.AMAZON]
        }

        with patch('dt_project.core.real_quantum_hardware_integration.IBMQuantumProvider') as mock_ibm:
            # Simulate IBM provider failure
            mock_ibm.side_effect = ProviderConnectionError("IBM Quantum service unavailable")

            # Test failover execution
            failover_result = await manager.execute_with_failover(job_config)

            assert failover_result.primary_provider_failed == True
            assert failover_result.failover_successful == True
            assert failover_result.used_provider in [QuantumProvider.GOOGLE, QuantumProvider.AMAZON]


class TestQuantumProviderImplementations:
    """Test specific quantum provider implementations."""

    @pytest.mark.asyncio
    async def test_ibm_quantum_provider(self):
        """Test IBM Quantum provider implementation."""

        credentials = QuantumCredentials(
            provider=QuantumProvider.IBM,
            api_token='fake_ibm_token',
            instance='h/g/p'
        )

        with patch('dt_project.core.real_quantum_hardware_integration.IBMQ') as mock_ibmq:
            mock_provider = Mock()
            mock_backend = Mock()
            mock_backend.name.return_value = 'ibm_brisbane'
            mock_backend.configuration.return_value.n_qubits = 127

            mock_provider.backends.return_value = [mock_backend]
            mock_ibmq.get_provider.return_value = mock_provider

            ibm_provider = IBMQuantumProvider(credentials)
            await ibm_provider.initialize()

            # Test device listing
            devices = await ibm_provider.get_available_devices()
            assert len(devices) > 0
            assert devices[0].device_name == 'ibm_brisbane'

            # Test job submission
            mock_circuit = Mock()
            job_result = await ibm_provider.submit_job(mock_circuit, 'ibm_brisbane', 1000)
            assert job_result.submission_successful == True

    @pytest.mark.asyncio
    async def test_amazon_braket_provider(self):
        """Test Amazon Braket provider implementation."""

        credentials = QuantumCredentials(
            provider=QuantumProvider.AMAZON,
            aws_access_key_id='fake_access_key',
            aws_secret_access_key='fake_secret_key',
            aws_region='us-east-1'
        )

        with patch('dt_project.core.real_quantum_hardware_integration.AwsDevice') as mock_device:
            mock_device_instance = Mock()
            mock_device_instance.name = 'IonQ Device'
            mock_device_instance.properties.paradigm.qubitCount = 32

            mock_device.get_devices.return_value = [mock_device_instance]

            braket_provider = AmazonBraketProvider(credentials)
            await braket_provider.initialize()

            # Test device discovery
            devices = await braket_provider.get_available_devices()
            assert len(devices) > 0

            # Test circuit execution
            mock_circuit = Mock()
            execution_result = await braket_provider.execute_circuit(
                mock_circuit, 'IonQ Device', 1000
            )
            assert execution_result.execution_successful == True

    @pytest.mark.asyncio
    async def test_google_quantum_provider(self):
        """Test Google Quantum AI provider implementation."""

        credentials = QuantumCredentials(
            provider=QuantumProvider.GOOGLE,
            service_account_key='{"type": "service_account"}',
            project_id='quantum-test-project'
        )

        with patch('dt_project.core.real_quantum_hardware_integration.cirq_google') as mock_cirq:
            mock_engine = Mock()
            mock_processor = Mock()
            mock_processor.processor_id = 'weber'

            mock_engine.list_processors.return_value = [mock_processor]
            mock_cirq.get_engine.return_value = mock_engine

            google_provider = GoogleQuantumProvider(credentials)
            await google_provider.initialize()

            # Test processor listing
            processors = await google_provider.get_available_processors()
            assert len(processors) > 0
            assert processors[0].device_name == 'weber'

    @pytest.mark.asyncio
    async def test_azure_quantum_provider(self):
        """Test Azure Quantum provider implementation."""

        credentials = QuantumCredentials(
            provider=QuantumProvider.AZURE,
            subscription_id='fake-subscription-id',
            resource_group='quantum-rg',
            workspace_name='quantum-workspace',
            location='East US'
        )

        with patch('dt_project.core.real_quantum_hardware_integration.Workspace') as mock_workspace:
            mock_workspace_instance = Mock()
            mock_target = Mock()
            mock_target.name = 'ionq.simulator'

            mock_workspace_instance.get_targets.return_value = [mock_target]
            mock_workspace.return_value = mock_workspace_instance

            azure_provider = AzureQuantumProvider(credentials)
            await azure_provider.initialize()

            # Test target discovery
            targets = await azure_provider.get_available_targets()
            assert len(targets) > 0


class TestQuantumHardwareErrorHandling:
    """Test error handling in quantum hardware operations."""

    def test_hardware_error_types(self):
        """Test different hardware error types."""

        # Test hardware error
        with pytest.raises(HardwareError):
            raise HardwareError("General hardware error")

        # Test provider connection error
        with pytest.raises(ProviderConnectionError):
            raise ProviderConnectionError("Failed to connect to quantum provider")

        # Test job execution error
        with pytest.raises(JobExecutionError):
            raise JobExecutionError("Quantum job execution failed")

        # Test credential error
        with pytest.raises(CredentialError):
            raise CredentialError("Invalid quantum provider credentials")

        # Test device unavailable error
        with pytest.raises(DeviceUnavailableError):
            raise DeviceUnavailableError("Quantum device is offline")

    @pytest.mark.asyncio
    async def test_connection_recovery_mechanisms(self):
        """Test connection recovery and retry mechanisms."""

        manager = QuantumHardwareManager({
            'max_retry_attempts': 3,
            'retry_delay_seconds': 1,
            'exponential_backoff': True
        })

        # Test connection with retries
        with patch('dt_project.core.real_quantum_hardware_integration.IBMQuantumProvider') as mock_provider:
            # Simulate intermittent connection failure
            mock_provider.side_effect = [
                ProviderConnectionError("Connection failed"),
                ProviderConnectionError("Connection failed"),
                Mock()  # Success on third attempt
            ]

            connection_result = await manager.connect_with_retry(QuantumProvider.IBM)
            assert connection_result.connection_successful == True
            assert connection_result.attempts_made == 3

    @pytest.mark.asyncio
    async def test_job_cancellation_and_cleanup(self):
        """Test job cancellation and resource cleanup."""

        job_config = {
            'job_id': 'cancellation_test_job',
            'timeout_minutes': 1
        }

        quantum_job = QuantumJob(job_config)

        # Test job cancellation
        cancellation_result = await quantum_job.cancel()
        assert cancellation_result.cancellation_successful == True
        assert quantum_job.status == JobStatus.CANCELLED

        # Test resource cleanup
        cleanup_result = await quantum_job.cleanup_resources()
        assert cleanup_result.cleanup_successful == True

    def test_device_capability_validation(self):
        """Test device capability validation before job submission."""

        device_specs = QuantumDeviceSpecs(
            provider=QuantumProvider.IBM,
            device_name='test_device',
            n_qubits=5,
            hardware_type=QuantumHardwareType.GATE_BASED,
            connectivity='linear',
            gate_fidelity=0.99,
            coherence_time_us=50.0,
            gate_time_ns=100.0
        )

        # Test valid circuit
        valid_circuit = {
            'qubits': 4,
            'gates': [{'type': 'h', 'qubit': 0}]
        }

        validation_result = device_specs.validate_circuit_compatibility(valid_circuit)
        assert validation_result.is_compatible == True

        # Test invalid circuit (too many qubits)
        invalid_circuit = {
            'qubits': 10,  # More than device supports
            'gates': [{'type': 'h', 'qubit': 0}]
        }

        validation_result = device_specs.validate_circuit_compatibility(invalid_circuit)
        assert validation_result.is_compatible == False
        assert 'insufficient_qubits' in validation_result.compatibility_issues


class TestQuantumResourceOptimization:
    """Test quantum resource optimization and management."""

    @pytest.mark.asyncio
    async def test_resource_allocation_optimization(self):
        """Test optimal resource allocation across quantum devices."""

        resource_manager = QuantumResourceManager({
            'max_concurrent_jobs': 5,
            'resource_balancing': 'load_aware',
            'cost_optimization': True
        })

        # Mock multiple job requests
        job_requests = []
        for i in range(10):
            job_requests.append({
                'job_id': f'resource_job_{i:03d}',
                'required_qubits': np.random.randint(2, 8),
                'shots': np.random.randint(100, 2000),
                'priority': 'normal' if i < 7 else 'high'
            })

        # Test resource allocation
        allocation_result = await resource_manager.allocate_resources(job_requests)

        assert allocation_result.allocation_successful == True
        assert len(allocation_result.allocated_jobs) <= 5  # Respects concurrency limit
        assert allocation_result.resource_utilization > 0.5

        # Verify high-priority jobs get preference
        allocated_priorities = [j['priority'] for j in allocation_result.allocated_jobs]
        high_priority_count = allocated_priorities.count('high')
        assert high_priority_count == min(3, len(allocation_result.allocated_jobs))

    @pytest.mark.asyncio
    async def test_cost_optimization_strategies(self):
        """Test cost optimization across quantum providers."""

        cost_optimizer = HardwareOptimization({
            'cost_weight': 0.4,
            'performance_weight': 0.4,
            'availability_weight': 0.2
        })

        # Mock provider pricing
        provider_costs = {
            QuantumProvider.IBM: {'per_shot': 0.001, 'queue_fee': 0.1},
            QuantumProvider.AMAZON: {'per_shot': 0.002, 'queue_fee': 0.05},
            QuantumProvider.GOOGLE: {'per_shot': 0.0015, 'queue_fee': 0.08}
        }

        job_requirements = {
            'shots': 1000,
            'required_qubits': 8,
            'max_cost': 5.0
        }

        # Test cost optimization
        optimization_result = await cost_optimizer.optimize_provider_selection(
            job_requirements, provider_costs
        )

        assert optimization_result.optimization_successful == True
        assert optimization_result.estimated_cost <= job_requirements['max_cost']
        assert optimization_result.recommended_provider is not None

    def test_performance_benchmarking(self):
        """Test quantum hardware performance benchmarking."""

        benchmark_suite = {
            'random_circuit': {'qubits': 5, 'depth': 10},
            'quantum_volume': {'qubits': 8},
            'process_tomography': {'qubits': 2},
            'gate_set_tomography': {'gates': ['h', 'cnot', 'rz']}
        }

        device_specs = QuantumDeviceSpecs(
            provider=QuantumProvider.IBM,
            device_name='benchmark_device',
            n_qubits=20,
            hardware_type=QuantumHardwareType.GATE_BASED,
            connectivity='heavy_hex',
            gate_fidelity=0.998,
            coherence_time_us=100.0,
            gate_time_ns=160.0
        )

        # Test benchmark execution
        benchmark_result = device_specs.run_benchmark_suite(benchmark_suite)

        assert benchmark_result.benchmark_successful == True
        assert 'random_circuit' in benchmark_result.benchmark_scores
        assert 'quantum_volume' in benchmark_result.benchmark_scores
        assert benchmark_result.overall_score > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])