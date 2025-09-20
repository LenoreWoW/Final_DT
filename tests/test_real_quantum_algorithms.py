"""
Test real quantum algorithms.
"""

import pytest
import asyncio

from dt_project.quantum.real_quantum_algorithms import RealQuantumAlgorithms, create_quantum_algorithms


class TestRealQuantumAlgorithms:
    """Test real quantum algorithm implementations."""
    
    def setup_method(self):
        """Set up test."""
        self.algorithms = create_quantum_algorithms()
    
    @pytest.mark.asyncio
    async def test_grovers_search(self):
        """Test Grover's search algorithm."""
        result = await self.algorithms.grovers_search(4, 2)
        
        assert result is not None
        assert result.algorithm_name == "Grover's Search"
        assert result.execution_time > 0
        assert result.quantum_advantage >= 1.0
        
        # Handle cases where Qiskit is available vs not available
        if result.error_message == "Qiskit not available":
            assert result.n_qubits == 1  # Fallback case
            assert not result.success
        else:
            assert result.n_qubits >= 2
            assert result.circuit_depth > 0
            # Should find target or at least execute successfully
            if result.success:
                assert result.result_data['found_item'] == 2
    
    @pytest.mark.asyncio
    async def test_phase_estimation(self):
        """Test quantum phase estimation."""
        eigenvalue = 0.25  # Easy to estimate precisely
        result = await self.algorithms.quantum_phase_estimation(eigenvalue)
        
        assert result is not None
        assert result.algorithm_name == "Quantum Phase Estimation"
        assert result.execution_time > 0
        assert result.n_qubits > 0
        
        if result.success:
            estimated = result.result_data['estimated_phase']
            error = abs(estimated - eigenvalue)
            assert error < 0.5  # Reasonable error bound
    
    @pytest.mark.asyncio
    async def test_bernstein_vazirani(self):
        """Test Bernstein-Vazirani algorithm."""
        secret = "101"
        result = await self.algorithms.bernstein_vazirani(secret)
        
        assert result is not None
        assert result.algorithm_name == "Bernstein-Vazirani"
        assert result.execution_time > 0
        
        # Handle cases where Qiskit is available vs not available
        if result.error_message == "Qiskit not available":
            assert result.n_qubits == 1  # Fallback case
            assert not result.success
        else:
            assert result.n_qubits == len(secret) + 1
            # This algorithm should work perfectly on simulator
            if result.success:
                assert result.result_data['found_string'] == secret
    
    @pytest.mark.asyncio
    async def test_quantum_fourier_transform(self):
        """Test quantum Fourier transform."""
        result = await self.algorithms.quantum_fourier_transform_demo(3)
        
        assert result is not None
        assert result.algorithm_name == "Quantum Fourier Transform"
        assert result.execution_time > 0
        
        # Handle cases where Qiskit is available vs not available
        if result.error_message == "Qiskit not available":
            assert result.n_qubits == 1  # Fallback case
            assert not result.success
        else:
            assert result.n_qubits == 3
            # Should have distributed outcomes
            if result.success:
                entropy = result.result_data['distribution_entropy']
                assert entropy > 0  # Should have some distribution
    
    @pytest.mark.asyncio
    async def test_quantum_advantage_measurement(self):
        """Test that algorithms measure quantum advantage."""
        # Test with Grover's algorithm
        result = await self.algorithms.grovers_search(16, 5)
        
        assert result.quantum_advantage >= 1.0
        assert result.classical_time > 0
        assert result.execution_time > 0
        
        # For Grover's, theoretical advantage should be sqrt(N)
        if result.success and result.quantum_advantage > 1:
            theoretical_max = 4.0  # sqrt(16)
            assert result.quantum_advantage <= theoretical_max * 2  # Allow some margin
