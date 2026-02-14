"""Unified configuration manager stub."""

import os, json

_instance = None

def _parse_bool(val):
    if isinstance(val, bool):
        return val
    return str(val).lower() in ('true', 'yes', '1', 'on')


class _Section:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class UnifiedConfigManager:
    def __init__(self, config_file=None, env_file=None):
        # Defaults
        self.environment = os.getenv('FLASK_ENV', 'development')
        self.debug = True
        self.port = 8000
        self.quantum = _Section(enabled=True, backend='aer_simulator', shots=1024,
                                 max_qubits=20, error_threshold=0.01)
        self.database = _Section(url='sqlite:///quantum_twins.db')
        self.features = _Section(enable_fault_tolerance=True)

        # Load config file
        if config_file and os.path.isfile(config_file):
            try:
                with open(config_file) as f:
                    data = json.load(f)
                defaults = data.get('default', {})
                env_data = data.get(self.environment, {})
                merged = {**defaults, **env_data}
                if 'debug' in merged:
                    self.debug = _parse_bool(merged['debug'])
                if 'port' in merged:
                    self.port = int(merged['port'])
                q = {**defaults.get('quantum', {}), **env_data.get('quantum', {})}
                if 'backend' in q:
                    self.quantum.backend = q['backend']
                if 'shots' in q:
                    self.quantum.shots = int(q['shots'])
            except Exception:
                pass

        # Load env file
        if env_file and os.path.isfile(env_file):
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            k, v = line.split('=', 1)
                            os.environ.setdefault(k.strip(), v.strip())
            except Exception:
                pass

        # Override from env vars
        if 'DEBUG' in os.environ:
            self.debug = _parse_bool(os.environ['DEBUG'])
        if 'PORT' in os.environ:
            self.port = int(os.environ['PORT'])
        if 'QUANTUM_BACKEND' in os.environ:
            self.quantum.backend = os.environ['QUANTUM_BACKEND']
        if 'QUANTUM_SHOTS' in os.environ:
            self.quantum.shots = int(os.environ['QUANTUM_SHOTS'])
        if 'QUANTUM_MAX_QUBITS' in os.environ:
            self.quantum.max_qubits = int(os.environ['QUANTUM_MAX_QUBITS'])
        if 'QUANTUM_ERROR_THRESHOLD' in os.environ:
            self.quantum.error_threshold = float(os.environ['QUANTUM_ERROR_THRESHOLD'])

        # Validation
        if self.quantum.max_qubits < 1:
            raise ValueError("max_qubits must be at least 1")
        if self.quantum.shots < 1:
            raise ValueError("shots must be at least 1")
        if not (0 <= self.quantum.error_threshold <= 1):
            raise ValueError("error_threshold must be between 0 and 1")

    def get_flask_config(self):
        return {
            'DEBUG': self.debug,
            'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret'),
            'SQLALCHEMY_DATABASE_URI': self.database.url,
            'SQLALCHEMY_ENGINE_OPTIONS': {'pool_pre_ping': True},
        }

    def get_quantum_config(self):
        return {
            'fault_tolerance': self.features.enable_fault_tolerance,
            'holographic_viz': False,
            'max_qubits': self.quantum.max_qubits,
            'backend': self.quantum.backend,
            'shots': self.quantum.shots,
        }

    def get_summary(self):
        return {
            'environment': self.environment,
            'quantum_enabled': self.quantum.enabled,
            'features_enabled': {'fault_tolerance': self.features.enable_fault_tolerance},
            'apis_configured': {'twins': True, 'benchmark': True, 'auth': True},
        }


def get_unified_config():
    global _instance
    if _instance is None:
        _instance = UnifiedConfigManager()
    return _instance


def reset_config():
    global _instance
    _instance = None
