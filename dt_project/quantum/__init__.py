"""
Quantum Computing Module
Implements quantum-enhanced simulations and machine learning for digital twin.
"""

from typing import Optional
import logging

# Import available modules only
try:
    from dt_project.quantum.quantum_digital_twin_core import QuantumDigitalTwinCore
    from dt_project.quantum.quantum_optimization import QuantumOptimizer
    from dt_project.quantum.quantum_ai_systems import QuantumNeuralNetwork
except ImportError as e:
    print(f"Warning: Some quantum modules not available: {e}")

logger = logging.getLogger(__name__)

def initialize_quantum_components(config=None) -> dict:
    """
    Initialize available quantum components.
    
    Returns:
        Dictionary with available quantum components
    """
    
    logger.info("Initializing quantum components...")
    
    available_components = {}
    
    try:
        available_components["quantum_digital_twin"] = QuantumDigitalTwinCore
        logger.info("✅ Quantum Digital Twin Core available")
    except NameError:
        logger.warning("❌ Quantum Digital Twin Core not available")
    
    try:
        available_components["quantum_optimizer"] = QuantumOptimizer
        logger.info("✅ Quantum Optimizer available")
    except NameError:
        logger.warning("❌ Quantum Optimizer not available")
    
    try:
        available_components["quantum_neural_network"] = QuantumNeuralNetwork  
        logger.info("✅ Quantum Neural Network available")
    except NameError:
        logger.warning("❌ Quantum Neural Network not available")
    
    return available_components

__all__ = ["initialize_quantum_components"] 