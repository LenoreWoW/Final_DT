"""
Unified Configuration Management
Consolidates all configuration sources into a single, coherent system.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class QuantumConfig:
    """Quantum-specific configuration."""
    enabled: bool = True
    backend: str = 'aer_simulator'
    shots: int = 1024
    max_qubits: int = 25
    error_threshold: float = 0.001
    ibmq_token: Optional[str] = None
    ibmq_provider: str = 'ibm-q'
    ibmq_backend: str = 'ibm_qasm_simulator'

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = 'sqlite:///quantum_platform.db'
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = field(default_factory=lambda: os.urandom(32).hex())
    rate_limit_per_minute: int = 60
    session_timeout: int = 3600
    cors_origins: list = field(default_factory=lambda: ['http://localhost:3000', 'http://localhost:8000'])

@dataclass
class FeatureConfig:
    """Feature toggles."""
    enable_quantum_internet: bool = True
    enable_fault_tolerance: bool = True
    enable_holographic_viz: bool = True
    enable_websocket: bool = True
    enable_graphql: bool = True
    enable_metrics: bool = True
    enable_profiling: bool = False

@dataclass
class APIConfig:
    """External API configuration."""
    weather_api_key: Optional[str] = None
    weather_base_url: str = 'https://api.open-meteo.com/v1/forecast'
    geocoding_api_key: Optional[str] = None
    geocoding_base_url: str = 'https://nominatim.openstreetmap.org'

class UnifiedConfigManager:
    """
    Unified configuration manager that consolidates all configuration sources.
    
    Priority order:
    1. Environment variables
    2. .env file
    3. config.json file
    4. Default values
    """
    
    def __init__(self, env_file: Optional[str] = None, config_file: Optional[str] = None):
        self.env_file = env_file or '.env'
        self.config_file = config_file or 'config/config.json'
        self.environment = os.getenv('FLASK_ENV', 'development')
        
        # Configuration sections
        self.quantum = QuantumConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.features = FeatureConfig()
        self.api = APIConfig()
        
        # Core settings
        self.debug = True
        self.port = 8000
        self.log_level = 'INFO'
        self.cache_timeout = 300
        
        # Load configuration
        self._load_configuration()
        
    def _load_configuration(self):
        """Load configuration from all sources."""
        try:
            # 1. Load from JSON config file
            self._load_from_json()
            
            # 2. Load from .env file
            self._load_from_env_file()
            
            # 3. Override with environment variables
            self._load_from_environment()
            
            # 4. Validate configuration
            self._validate_configuration()
            
            logger.info(f"Configuration loaded for environment: {self.environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise  # Re-raise for validation errors
            
    def _load_from_json(self):
        """Load configuration from JSON file."""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                
                # Load default config
                default_config = config_data.get('default', {})
                self._apply_config_section(default_config)
                
                # Load environment-specific config
                env_config = config_data.get(self.environment, {})
                self._apply_config_section(env_config)
                
        except Exception as e:
            logger.warning(f"Could not load JSON config: {e}")
    
    def _load_from_env_file(self):
        """Load configuration from .env file."""
        try:
            env_path = Path(self.env_file)
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Core settings
        self.debug = self._get_bool_env('DEBUG', self.debug)
        self.port = self._get_int_env('PORT', self.port)
        self.log_level = os.getenv('LOG_LEVEL', self.log_level)
        self.cache_timeout = self._get_int_env('CACHE_TIMEOUT', self.cache_timeout)
        
        # Quantum configuration
        self.quantum.enabled = self._get_bool_env('QUANTUM_ENABLED', self.quantum.enabled)
        self.quantum.backend = os.getenv('QUANTUM_BACKEND', self.quantum.backend)
        self.quantum.shots = self._get_int_env('QUANTUM_SHOTS', self.quantum.shots)
        self.quantum.max_qubits = self._get_int_env('QUANTUM_MAX_QUBITS', self.quantum.max_qubits)
        self.quantum.error_threshold = self._get_float_env('QUANTUM_ERROR_THRESHOLD', self.quantum.error_threshold)
        self.quantum.ibmq_token = os.getenv('IBMQ_TOKEN')
        self.quantum.ibmq_provider = os.getenv('IBMQ_PROVIDER', self.quantum.ibmq_provider)
        self.quantum.ibmq_backend = os.getenv('IBMQ_BACKEND', self.quantum.ibmq_backend)
        
        # Database configuration
        self.database.url = os.getenv('DATABASE_URL', self.database.url)
        self.database.pool_size = self._get_int_env('DATABASE_POOL_SIZE', self.database.pool_size)
        self.database.max_overflow = self._get_int_env('DATABASE_MAX_OVERFLOW', self.database.max_overflow)
        self.database.pool_timeout = self._get_int_env('DATABASE_POOL_TIMEOUT', self.database.pool_timeout)
        
        # Security configuration
        self.security.secret_key = os.getenv('SECRET_KEY', self.security.secret_key)
        self.security.rate_limit_per_minute = self._get_int_env('RATE_LIMIT_PER_MINUTE', self.security.rate_limit_per_minute)
        self.security.session_timeout = self._get_int_env('SESSION_TIMEOUT', self.security.session_timeout)
        cors_origins = os.getenv('CORS_ORIGINS')
        if cors_origins:
            self.security.cors_origins = [origin.strip() for origin in cors_origins.split(',')]
        
        # Feature configuration
        self.features.enable_quantum_internet = self._get_bool_env('ENABLE_QUANTUM_INTERNET', self.features.enable_quantum_internet)
        self.features.enable_fault_tolerance = self._get_bool_env('ENABLE_FAULT_TOLERANCE', self.features.enable_fault_tolerance)
        self.features.enable_holographic_viz = self._get_bool_env('ENABLE_HOLOGRAPHIC_VIZ', self.features.enable_holographic_viz)
        self.features.enable_websocket = self._get_bool_env('ENABLE_WEBSOCKET', self.features.enable_websocket)
        self.features.enable_graphql = self._get_bool_env('ENABLE_GRAPHQL', self.features.enable_graphql)
        self.features.enable_metrics = self._get_bool_env('ENABLE_METRICS', self.features.enable_metrics)
        self.features.enable_profiling = self._get_bool_env('ENABLE_PROFILING', self.features.enable_profiling)
        
        # API configuration
        self.api.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.api.weather_base_url = os.getenv('WEATHER_BASE_URL', self.api.weather_base_url)
        self.api.geocoding_api_key = os.getenv('GEOCODING_API_KEY')
        self.api.geocoding_base_url = os.getenv('GEOCODING_BASE_URL', self.api.geocoding_base_url)
    
    def _apply_config_section(self, config_section: Dict[str, Any]):
        """Apply a configuration section to the appropriate config objects."""
        # Core settings
        if 'debug' in config_section:
            self.debug = config_section['debug']
        if 'port' in config_section:
            self.port = config_section['port']
        if 'log_level' in config_section:
            self.log_level = config_section['log_level']
        
        # Quantum settings
        quantum_config = config_section.get('quantum', {})
        for key, value in quantum_config.items():
            if hasattr(self.quantum, key):
                setattr(self.quantum, key, value)
        
        # Database settings
        database_config = config_section.get('database', {})
        for key, value in database_config.items():
            if hasattr(self.database, key):
                setattr(self.database, key, value)
        
        # Feature settings
        features_config = config_section.get('features', {})
        for key, value in features_config.items():
            if hasattr(self.features, key):
                setattr(self.features, key, value)
        
        # API settings
        api_config = config_section.get('api', {})
        for key, value in api_config.items():
            if hasattr(self.api, key):
                setattr(self.api, key, value)
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', 'yes', '1', 'on')
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}")
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}")
            return default
    
    def _validate_configuration(self):
        """Validate configuration values."""
        # Validate quantum configuration
        if self.quantum.max_qubits < 1:
            raise ValueError("max_qubits must be at least 1")
        if self.quantum.shots < 1:
            raise ValueError("shots must be at least 1")
        if not 0 < self.quantum.error_threshold < 1:
            raise ValueError("error_threshold must be between 0 and 1")
        
        # Validate database configuration
        if not self.database.url:
            raise ValueError("Database URL is required")
        
        # Validate security configuration
        if len(self.security.secret_key) < 32:
            logger.warning("Secret key should be at least 32 characters for security")
        
        # Check required API keys if features are enabled
        if self.quantum.enabled and self.quantum.ibmq_token:
            if len(self.quantum.ibmq_token) < 10:
                logger.warning("IBMQ token appears to be invalid")
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask-compatible configuration dictionary."""
        return {
            'DEBUG': self.debug,
            'SECRET_KEY': self.security.secret_key,
            'SQLALCHEMY_DATABASE_URI': self.database.url,
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
            }
        }
    
    def get_quantum_config(self) -> Dict[str, Any]:
        """Get quantum-specific configuration."""
        return {
            'fault_tolerance': self.features.enable_fault_tolerance,
            'quantum_internet': self.features.enable_quantum_internet,
            'holographic_viz': self.features.enable_holographic_viz,
            'max_qubits': self.quantum.max_qubits,
            'error_threshold': self.quantum.error_threshold,
            'backend': self.quantum.backend,
            'shots': self.quantum.shots,
            'ibmq_token': self.quantum.ibmq_token,
            'ibmq_provider': self.quantum.ibmq_provider,
            'ibmq_backend': self.quantum.ibmq_backend
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == 'development'
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for diagnostics."""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'port': self.port,
            'quantum_enabled': self.quantum.enabled,
            'quantum_backend': self.quantum.backend,
            'database_type': 'sqlite' if 'sqlite' in self.database.url else 'other',
            'features_enabled': {
                'quantum_internet': self.features.enable_quantum_internet,
                'fault_tolerance': self.features.enable_fault_tolerance,
                'websocket': self.features.enable_websocket,
                'graphql': self.features.enable_graphql,
                'metrics': self.features.enable_metrics,
            },
            'apis_configured': {
                'weather': self.api.weather_api_key is not None,
                'geocoding': self.api.geocoding_api_key is not None,
                'ibmq': self.quantum.ibmq_token is not None,
            }
        }

# Global configuration instance
_config_instance: Optional[UnifiedConfigManager] = None

def get_unified_config() -> UnifiedConfigManager:
    """Get the global unified configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = UnifiedConfigManager()
    return _config_instance

def reset_config():
    """Reset the global configuration instance (mainly for testing)."""
    global _config_instance
    _config_instance = None
