"""
Quantum Machine Learning
========================

Quantum ML algorithms and neural-quantum hybrid systems.
"""

# Try to import PennyLane-based modules (graceful degradation if PennyLane unavailable)
try:
    from .pennylane_quantum_ml import *
except (ImportError, AttributeError) as e:
    import logging
    logging.warning(f"PennyLane quantum ML not available: {e}")

try:
    from .neural_quantum_digital_twin import *
except (ImportError, AttributeError) as e:
    import logging
    logging.warning(f"Neural quantum digital twin not available: {e}")

try:
    from .enhanced_quantum_digital_twin import *
except (ImportError, AttributeError) as e:
    import logging
    logging.warning(f"Enhanced quantum digital twin not available: {e}")
