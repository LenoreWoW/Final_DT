"""
Test quantum digital twin core functionality.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch

from dt_project.quantum.quantum_digital_twin_core import (
    QuantumDigitalTwinCore, 
    QuantumTwinType,
    QuantumState,
    create_quantum_digital_twin_platform
)


class TestQuantumDigitalTwinCore:
    """Test the quantum digital twin core functionality."""
    
    def test_platform_creation(self):
        """Test creating quantum platform."""
        config = {
            'fault_tolerance': True,
            'quantum_internet': True,
            'holographic_viz': True,
            'max_qubits': 4,
            'error_threshold': 0.001
        }
        
        platform = create_quantum_digital_twin_platform(config)
        
        assert platform is not None
        assert hasattr(platform, 'quantum_network')
        assert hasattr(platform, 'quantum_sensors')
        assert hasattr(platform, 'quantum_ml')
        assert platform.fault_tolerance_enabled == True
        
    @pytest.mark.asyncio
    async def test_quantum_twin_creation(self):
        """Test creating a quantum digital twin."""
        config = {
            'fault_tolerance': False,  # Disable for testing
            'quantum_internet': False,
            'holographic_viz': False,
            'max_qubits': 4
        }
        
        platform = QuantumDigitalTwinCore(config)
        
        initial_state = {
            'fitness_level': 0.8,
            'fatigue_level': 0.2,
            'technique_efficiency': 0.9
        }
        
        quantum_resources = {
            'n_qubits': 4,
            'circuit_depth': 10,
            'quantum_volume': 16,
            'error_threshold': 0.01
        }
        
        twin = await platform.create_quantum_digital_twin(
            entity_id="test_athlete",
            twin_type=QuantumTwinType.ATHLETE,
            initial_state=initial_state,
            quantum_resources=quantum_resources
        )
        
        assert twin is not None
        assert twin.entity_id == "test_athlete"
        assert twin.twin_type == QuantumTwinType.ATHLETE
        assert twin.quantum_state is not None
        assert twin.quantum_state.fidelity > 0.9
        
    def test_quantum_state_creation(self):
        """Test creating quantum state."""
        state_vector = np.array([1.0, 0.0, 0.0, 0.0])
        
        quantum_state = QuantumState(
            entity_id="test",
            state_vector=state_vector,
            coherence_time=1000.0,
            fidelity=0.99
        )
        
        assert quantum_state.entity_id == "test"
        assert np.allclose(quantum_state.state_vector, state_vector)
        assert quantum_state.fidelity == 0.99
        assert quantum_state.coherence_time == 1000.0
        
    @pytest.mark.asyncio
    async def test_classical_to_quantum_encoding(self):
        """Test encoding classical data to quantum state."""
        config = {'max_qubits': 4}
        platform = QuantumDigitalTwinCore(config)
        
        classical_data = {
            'value1': 0.5,
            'value2': 0.7,
            'value3': 0.3,
            'value4': 0.9
        }
        
        quantum_state = await platform._encode_classical_to_quantum(classical_data, 2)
        
        assert quantum_state is not None
        assert len(quantum_state.state_vector) == 4  # 2^2 qubits
        assert np.isclose(np.linalg.norm(quantum_state.state_vector), 1.0)  # Normalized
        
    def test_platform_summary(self):
        """Test getting platform summary."""
        config = {'max_qubits': 4}
        platform = QuantumDigitalTwinCore(config)
        
        summary = platform.get_quantum_advantage_summary()
        
        assert 'platform_status' in summary
        assert 'total_quantum_twins' in summary
        assert 'fault_tolerance_enabled' in summary
        assert summary['total_quantum_twins'] == 0  # No twins created yet
        
    @pytest.mark.asyncio
    async def test_quantum_evolution_basic(self):
        """Test basic quantum evolution without hanging."""
        config = {
            'fault_tolerance': False,
            'quantum_internet': False,
            'holographic_viz': False
        }
        platform = QuantumDigitalTwinCore(config)
        
        # Create a simple twin
        initial_state = {'test_value': 0.5}
        quantum_resources = {'n_qubits': 2, 'circuit_depth': 2}
        
        twin = await platform.create_quantum_digital_twin(
            "test_twin",
            QuantumTwinType.SYSTEM,
            initial_state,
            quantum_resources
        )
        
        # Test short evolution
        result = await platform.run_quantum_evolution("test_twin", 0.001)
        
        assert result is not None
        assert 'twin_id' in result
        assert 'quantum_fidelity' in result
        assert result['twin_id'] == "test_twin"
