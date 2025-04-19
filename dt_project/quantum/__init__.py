"""
Quantum Computing Module
Implements quantum-enhanced simulations and machine learning for digital twin.
"""

from typing import Optional
import logging

from dt_project.config import ConfigManager
from dt_project.quantum.qmc import QuantumMonteCarlo
from dt_project.quantum.qml import QuantumML
from dt_project.quantum.classical_mc import ClassicalMonteCarlo

logger = logging.getLogger(__name__)

def initialize_quantum_components(config: Optional[ConfigManager] = None) -> dict:
    """
    Initialize all quantum components with the provided configuration.
    
    Args:
        config: Configuration manager. If None, creates a new one.
        
    Returns:
        Dictionary with initialized quantum components
    """
    config = config or ConfigManager()
    
    # Initialize quantum components
    quantum_monte_carlo = QuantumMonteCarlo(config)
    quantum_ml = QuantumML(config)
    classical_monte_carlo = ClassicalMonteCarlo(config)
    
    # Check availability
    qmc_available = quantum_monte_carlo.is_available()
    qml_available = quantum_ml.is_available()
    
    if qmc_available:
        logger.info("Quantum Monte Carlo simulation is available")
    else:
        logger.warning("Quantum Monte Carlo simulation is not available, using classical fallback")
        
    if qml_available:
        logger.info("Quantum Machine Learning is available")
    else:
        logger.warning("Quantum Machine Learning is not available, using classical fallback")
    
    return {
        "monte_carlo": quantum_monte_carlo,
        "machine_learning": quantum_ml,
        "classical_monte_carlo": classical_monte_carlo,
        "qmc_available": qmc_available,
        "qml_available": qml_available
    }

__all__ = ["QuantumMonteCarlo", "QuantumML", "ClassicalMonteCarlo", "initialize_quantum_components"] 