#!/usr/bin/env python3
"""
🌌 QUANTUM DIGITAL TWIN PLATFORM
================================

Complete quantum computing platform with proven quantum advantages.

Organized Structure:
- core/         : Core quantum infrastructure
- algorithms/   : Quantum algorithms (QAOA, Sensing, Optimization)
- ml/           : Quantum machine learning
- hardware/     : Real quantum hardware integration
- tensor_networks/ : Tensor network algorithms
- visualization/: Quantum visualization tools

Author: Hassan Al-Sahli
Version: 2.1.0 - Reorganized Structure with Backward Compatibility
"""

import sys
import logging

__version__ = "2.1.0"
__author__ = "Hassan Al-Sahli"

logger = logging.getLogger(__name__)

# =============================================================================
# Core quantum functionality
# =============================================================================
try:
    from .core import *
    from .core.quantum_digital_twin_core import QuantumDigitalTwinCore
    from .core.framework_comparison import QuantumFrameworkComparator, FrameworkType, AlgorithmType
    from .core.async_quantum_backend import AsyncQuantumBackend
    from .core.distributed_quantum_system import (
        DistributedQuantumSystem,
        DistributedQuantumSystemConfig,
        TaskPriority,
        create_distributed_quantum_system
    )
except (ImportError, AttributeError) as e:
    logger.debug(f"Quantum core not fully available: {e}")

# =============================================================================
# Quantum algorithms
# =============================================================================
try:
    from .algorithms import *
    from .algorithms.quantum_sensing_digital_twin import (
        QuantumSensingDigitalTwin,
        QuantumSensingTheory,
        SensingModality,
        PrecisionScaling,
        SensingResult
    )
    from .algorithms.qaoa_optimizer import QAOAOptimizer, QAOAConfig, QAOAResult
    from .algorithms.quantum_optimization import QuantumOptimizationDigitalTwin
    from .algorithms.uncertainty_quantification import (
        UncertaintyQuantificationFramework,
        VirtualQPU,
        VirtualQPUConfig,
        UncertaintyType,
        UQResult
    )
    from .algorithms.proven_quantum_advantage import ProvenQuantumAdvantageValidator
    from .algorithms.real_quantum_algorithms import RealQuantumAlgorithms
    from .algorithms.error_matrix_digital_twin import (
        ErrorMatrixDigitalTwin,
        ErrorMatrixResult,
        ErrorCharacterization,
        ErrorType,
        create_error_matrix_twin
    )
    from .algorithms.quantum_sensing_digital_twin import create_quantum_sensing_twin
    from .algorithms.qaoa_optimizer import create_maxcut_qaoa
    from .algorithms.uncertainty_quantification import create_uq_framework
except (ImportError, AttributeError) as e:
    logger.debug(f"Quantum algorithms not fully available: {e}")

# =============================================================================
# Quantum ML
# =============================================================================
try:
    from .ml import *
    from .ml.pennylane_quantum_ml import PennyLaneQuantumML, create_quantum_ml_classifier
    from .ml.neural_quantum_digital_twin import NeuralQuantumDigitalTwin, create_neural_quantum_twin
    from .ml.enhanced_quantum_digital_twin import EnhancedQuantumDigitalTwin
except (ImportError, AttributeError) as e:
    logger.debug(f"Quantum ML not fully available: {e}")

# =============================================================================
# Hardware integration
# =============================================================================
try:
    from .hardware import *
    from .hardware.real_hardware_backend import RealHardwareBackend
    from .hardware.nisq_hardware_integration import (
        NISQHardwareIntegration,
        NISQHardwareIntegrator,
        create_nisq_integrator,
        NISQConfig
    )
except (ImportError, AttributeError) as e:
    logger.debug(f"Quantum hardware not fully available: {e}")

# =============================================================================
# Tensor networks
# =============================================================================
try:
    from .tensor_networks import *
    from .tensor_networks.matrix_product_operator import MatrixProductOperator
    from .tensor_networks.tree_tensor_network import TreeTensorNetwork, TTNConfig, create_ttn_for_benchmarking
except (ImportError, AttributeError) as e:
    logger.debug(f"Tensor networks not fully available: {e}")

# =============================================================================
# Visualization
# =============================================================================
try:
    from .visualization import *
except (ImportError, AttributeError) as e:
    logger.debug(f"Visualization not fully available: {e}")

# =============================================================================
# BACKWARD COMPATIBILITY: Create module-level aliases
# These allow imports like: from dt_project.quantum.quantum_sensing_digital_twin import ...
# =============================================================================
try:
    from .algorithms import quantum_sensing_digital_twin
    sys.modules['dt_project.quantum.quantum_sensing_digital_twin'] = quantum_sensing_digital_twin
except (ImportError, AttributeError):
    pass

try:
    from .algorithms import qaoa_optimizer
    sys.modules['dt_project.quantum.qaoa_optimizer'] = qaoa_optimizer
except (ImportError, AttributeError):
    pass

try:
    from .algorithms import quantum_optimization
    sys.modules['dt_project.quantum.quantum_optimization'] = quantum_optimization
except (ImportError, AttributeError):
    pass

try:
    from .algorithms import uncertainty_quantification
    sys.modules['dt_project.quantum.uncertainty_quantification'] = uncertainty_quantification
except (ImportError, AttributeError):
    pass

try:
    from .algorithms import proven_quantum_advantage
    sys.modules['dt_project.quantum.proven_quantum_advantage'] = proven_quantum_advantage
except (ImportError, AttributeError):
    pass

try:
    from .algorithms import real_quantum_algorithms
    sys.modules['dt_project.quantum.real_quantum_algorithms'] = real_quantum_algorithms
except (ImportError, AttributeError):
    pass

try:
    from .algorithms import error_matrix_digital_twin
    sys.modules['dt_project.quantum.error_matrix_digital_twin'] = error_matrix_digital_twin
except (ImportError, AttributeError):
    pass

try:
    from .core import quantum_digital_twin_core
    sys.modules['dt_project.quantum.quantum_digital_twin_core'] = quantum_digital_twin_core
except (ImportError, AttributeError):
    pass

try:
    from .core import framework_comparison
    sys.modules['dt_project.quantum.framework_comparison'] = framework_comparison
except (ImportError, AttributeError):
    pass

try:
    from .core import async_quantum_backend
    sys.modules['dt_project.quantum.async_quantum_backend'] = async_quantum_backend
except (ImportError, AttributeError):
    pass

try:
    from .core import distributed_quantum_system
    sys.modules['dt_project.quantum.distributed_quantum_system'] = distributed_quantum_system
except (ImportError, AttributeError):
    pass

try:
    from .ml import pennylane_quantum_ml
    sys.modules['dt_project.quantum.pennylane_quantum_ml'] = pennylane_quantum_ml
except (ImportError, AttributeError):
    pass

try:
    from .ml import neural_quantum_digital_twin
    sys.modules['dt_project.quantum.neural_quantum_digital_twin'] = neural_quantum_digital_twin
except (ImportError, AttributeError):
    pass

try:
    from .ml import enhanced_quantum_digital_twin
    sys.modules['dt_project.quantum.enhanced_quantum_digital_twin'] = enhanced_quantum_digital_twin
except (ImportError, AttributeError):
    pass

try:
    from .hardware import nisq_hardware_integration
    sys.modules['dt_project.quantum.nisq_hardware_integration'] = nisq_hardware_integration
except (ImportError, AttributeError):
    pass

try:
    from .hardware import real_hardware_backend
    sys.modules['dt_project.quantum.real_hardware_backend'] = real_hardware_backend
except (ImportError, AttributeError):
    pass

try:
    from .tensor_networks import tree_tensor_network
    sys.modules['dt_project.quantum.tree_tensor_network'] = tree_tensor_network
except (ImportError, AttributeError):
    pass

try:
    from .tensor_networks import matrix_product_operator
    sys.modules['dt_project.quantum.matrix_product_operator'] = matrix_product_operator
except (ImportError, AttributeError):
    pass

__all__ = [
    # Core
    'QuantumDigitalTwinCore',
    'QuantumFrameworkComparator',
    'FrameworkType',
    'AlgorithmType',
    'AsyncQuantumBackend',
    'DistributedQuantumSystem',
    'DistributedQuantumSystemConfig',
    'TaskPriority',
    'create_distributed_quantum_system',
    
    # Algorithms
    'QuantumSensingDigitalTwin',
    'QuantumSensingTheory',
    'SensingModality',
    'PrecisionScaling',
    'SensingResult',
    'create_quantum_sensing_twin',
    'QAOAOptimizer',
    'QAOAConfig',
    'QAOAResult',
    'create_maxcut_qaoa',
    'QuantumOptimizationDigitalTwin',
    'UncertaintyQuantificationFramework',
    'VirtualQPU',
    'VirtualQPUConfig',
    'UncertaintyType',
    'UQResult',
    'create_uq_framework',
    'ProvenQuantumAdvantageValidator',
    'RealQuantumAlgorithms',
    'ErrorMatrixDigitalTwin',
    'ErrorMatrixResult',
    'ErrorCharacterization',
    'ErrorType',
    'create_error_matrix_twin',
    
    # ML
    'PennyLaneQuantumML',
    'create_quantum_ml_classifier',
    'NeuralQuantumDigitalTwin',
    'create_neural_quantum_twin',
    'EnhancedQuantumDigitalTwin',
    
    # Hardware
    'RealHardwareBackend',
    'NISQHardwareIntegration',
    'NISQHardwareIntegrator',
    'NISQConfig',
    'create_nisq_integrator',
    
    # Tensor Networks
    'TreeTensorNetwork',
    'TTNConfig',
    'create_ttn_for_benchmarking',
    'MatrixProductOperator',
]
