"""Config manager stub."""

import os

class ConfigManager:
    def __init__(self, config_file=None):
        self.config = {
            'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret'),
            'DATABASE_URL': os.getenv('DATABASE_URL', 'sqlite:///quantum_twins.db'),
            'QUANTUM_BACKEND': 'aer_simulator',
            'DEBUG': True,
        }

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config
