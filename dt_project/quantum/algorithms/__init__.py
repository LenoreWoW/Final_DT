"""
Quantum Algorithms
==================

Specialized quantum algorithms for optimization, sensing, and analysis.
"""

from .qaoa_optimizer import *
from .quantum_sensing_digital_twin import *
from .quantum_optimization import *
from .uncertainty_quantification import *
from .proven_quantum_advantage import *
from .real_quantum_algorithms import *
from .error_matrix_digital_twin import *

# Explicit exports for factory functions
__all__ = [
    # QAOA
    'QAOAOptimizer', 'QAOAConfig', 'QAOAResult', 'create_maxcut_qaoa',
    # Quantum Sensing
    'QuantumSensingDigitalTwin', 'QuantumSensingTheory', 'SensingModality', 
    'PrecisionScaling', 'SensingResult', 'create_quantum_sensing_twin',
    # Uncertainty Quantification
    'UncertaintyQuantificationFramework', 'VirtualQPU', 'VirtualQPUConfig',
    'UQResult', 'NoiseParameters', 'UncertaintyMetrics', 'create_uq_framework',
    # Error Matrix
    'ErrorMatrixDigitalTwin', 'ErrorMatrixResult', 'ErrorCharacterization',
    'ErrorType', 'create_error_matrix_twin',
    # Proven Quantum Advantage
    'ProvenQuantumAdvantage',
    # Real Quantum Algorithms
    'RealQuantumAlgorithms',
]
