"""dt_project.quantum - Quantum module shims."""


def get_platform_status():
    """Return platform status dict."""
    return {
        "status": "operational",
        "quantum_modules": 11,
        "quantum_ready": True,
        "backend": "aer_simulator",
        "version": "2.0.0",
    }
