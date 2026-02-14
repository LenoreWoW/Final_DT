"""Secure config stub."""

import os

def get_config():
    return {
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret'),
        'QUANTUM_BACKEND': 'aer_simulator',
        'DATABASE_URL': os.getenv('DATABASE_URL', 'sqlite:///quantum_twins.db'),
    }
