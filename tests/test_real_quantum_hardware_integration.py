"""
Tests for quantum hardware integration - Aer Simulator backend.
Tests the refactored interface that uses Qiskit Aer simulators.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from dt_project.core.real_quantum_hardware_integration import (
    QuantumProvider,
    QuantumHardwareType,
    QuantumDeviceSpecs,
    QuantumJobResult,
    QuantumCredentialsManager,
    AerSimulatorConnector,
    QuantumHardwareOrchestrator,
)


class TestQuantumProvider:
    """Test the QuantumProvider enum."""

    def test_aer_simulator_provider_exists(self):
        assert QuantumProvider.AER_SIMULATOR.value == "aer_simulator"

    def test_no_ibm_provider(self):
        """Verify IBM provider has been removed."""
        provider_values = [p.value for p in QuantumProvider]
        assert "ibm" not in provider_values
        assert "ibm_quantum" not in provider_values


class TestQuantumDeviceSpecs:
    """Test quantum device specifications."""

    def test_aer_device_specs(self):
        specs = QuantumDeviceSpecs(
            provider=QuantumProvider.AER_SIMULATOR,
            device_name="aer_simulator",
            n_qubits=30,
            hardware_type=QuantumHardwareType.GATE_BASED,
            connectivity="full",
            gate_fidelity=1.0,
            coherence_time_us=float("inf"),
            gate_time_ns=0,
            availability=True,
            queue_depth=0,
            cost_per_shot=0.0,
        )

        assert specs.provider == QuantumProvider.AER_SIMULATOR
        assert specs.device_name == "aer_simulator"
        assert specs.n_qubits == 30
        assert specs.gate_fidelity == 1.0
        assert specs.cost_per_shot == 0.0


class TestQuantumJobResult:
    """Test quantum job result dataclass."""

    def test_successful_result(self):
        result = QuantumJobResult(
            job_id="test-001",
            device="aer_simulator",
            provider=QuantumProvider.AER_SIMULATOR,
            status="completed",
            counts={"00": 500, "11": 500},
            execution_time=0.1,
            queue_time=0,
            cost=0.0,
            metadata={"transpiled_depth": 2},
        )
        assert result.status == "completed"
        assert result.counts is not None
        assert result.error is None

    def test_failed_result(self):
        result = QuantumJobResult(
            job_id="error",
            device="aer_simulator",
            provider=QuantumProvider.AER_SIMULATOR,
            status="failed",
            counts=None,
            execution_time=0.0,
            queue_time=0,
            cost=0,
            metadata={},
            error="Circuit execution failed",
        )
        assert result.status == "failed"
        assert result.error is not None


class TestQuantumCredentialsManager:
    """Test credential management."""

    def test_load_credentials_from_env(self):
        with patch.dict(
            "os.environ",
            {"AWS_ACCESS_KEY_ID": "test_key", "AWS_REGION": "us-east-1"},
        ):
            manager = QuantumCredentialsManager()
            creds = manager.get_credentials(QuantumProvider.AMAZON)
            assert creds.get("access_key") == "test_key"
            assert creds.get("region") == "us-east-1"

    def test_empty_credentials(self):
        manager = QuantumCredentialsManager()
        creds = manager.get_credentials(QuantumProvider.AER_SIMULATOR)
        assert creds == {}


class TestAerSimulatorConnector:
    """Test Aer simulator connection and execution."""

    def test_connector_creation(self):
        connector = AerSimulatorConnector()
        assert connector.simulator is None
        assert "aer_simulator" in connector.available_backends

    @pytest.mark.asyncio
    async def test_connect(self):
        connector = AerSimulatorConnector()
        # Mock AER_AVAILABLE
        with patch(
            "dt_project.core.real_quantum_hardware_integration.AER_AVAILABLE", True
        ):
            with patch(
                "dt_project.core.real_quantum_hardware_integration.AerSimulator"
            ) as mock_aer:
                mock_aer.return_value = Mock()
                result = await connector.connect()
                assert result is True
                assert connector.simulator is not None

    def test_get_device_specs(self):
        connector = AerSimulatorConnector()
        specs = connector.get_device_specs()
        assert specs is not None
        assert specs.provider == QuantumProvider.AER_SIMULATOR
        assert specs.n_qubits == 30
        assert specs.gate_fidelity == 1.0
        assert specs.cost_per_shot == 0.0

    @pytest.mark.asyncio
    async def test_execute_circuit(self):
        connector = AerSimulatorConnector()

        # Create mock simulator and job
        mock_job = Mock()
        mock_job.job_id.return_value = "test-job-123"
        mock_result = Mock()
        mock_result.get_counts.return_value = {"00": 500, "11": 500}
        mock_job.result.return_value = mock_result

        mock_simulator = Mock()
        mock_simulator.run.return_value = mock_job
        connector.simulator = mock_simulator

        # Create mock circuit
        mock_circuit = Mock()

        with patch(
            "dt_project.core.real_quantum_hardware_integration.transpile"
        ) as mock_transpile:
            mock_transpiled = Mock()
            mock_transpiled.depth.return_value = 2
            mock_transpile.return_value = mock_transpiled

            result = await connector.execute_circuit(mock_circuit, shots=1000)

            assert result.status == "completed"
            assert result.counts == {"00": 500, "11": 500}
            assert result.provider == QuantumProvider.AER_SIMULATOR


class TestQuantumHardwareOrchestrator:
    """Test the hardware orchestrator."""

    def test_orchestrator_creation(self):
        orchestrator = QuantumHardwareOrchestrator()
        assert orchestrator.connectors == {}
        assert orchestrator.job_history == []

    @pytest.mark.asyncio
    async def test_initialize_aer_provider(self):
        orchestrator = QuantumHardwareOrchestrator()

        with patch(
            "dt_project.core.real_quantum_hardware_integration.AER_AVAILABLE", True
        ):
            with patch(
                "dt_project.core.real_quantum_hardware_integration.AerSimulator"
            ) as mock_aer:
                mock_aer.return_value = Mock()
                results = await orchestrator.initialize_all_providers()

                assert QuantumProvider.AER_SIMULATOR in results
                assert results[QuantumProvider.AER_SIMULATOR] is True

    def test_execution_statistics_empty(self):
        orchestrator = QuantumHardwareOrchestrator()
        stats = orchestrator.get_execution_statistics()
        assert stats == {}

    def test_execution_statistics_with_history(self):
        orchestrator = QuantumHardwareOrchestrator()
        orchestrator.job_history = [
            QuantumJobResult(
                job_id="job-1",
                device="aer_simulator",
                provider=QuantumProvider.AER_SIMULATOR,
                status="completed",
                counts={"0": 1000},
                execution_time=0.5,
                queue_time=0,
                cost=0.0,
                metadata={},
            ),
            QuantumJobResult(
                job_id="job-2",
                device="aer_simulator",
                provider=QuantumProvider.AER_SIMULATOR,
                status="completed",
                counts={"0": 500, "1": 500},
                execution_time=0.3,
                queue_time=0,
                cost=0.0,
                metadata={},
            ),
        ]

        stats = orchestrator.get_execution_statistics()
        assert stats["total_jobs"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["total_cost_usd"] == 0.0
        assert "aer_simulator" in stats["provider_statistics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
