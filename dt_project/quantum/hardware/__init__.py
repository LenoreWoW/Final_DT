"""
Quantum Hardware Integration
============================

Real quantum hardware backends and NISQ device integration.
"""

# Try to import hardware modules (graceful degradation if dependencies unavailable)
try:
    from .real_hardware_backend import *
except (ImportError, AttributeError) as e:
    import logging
    logging.warning(f"Real hardware backend not available: {e}")

try:
    from .nisq_hardware_integration import *
except (ImportError, AttributeError) as e:
    import logging
    logging.warning(f"NISQ hardware integration not available: {e}")
