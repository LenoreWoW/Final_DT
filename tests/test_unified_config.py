"""
Test unified configuration management.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch

from dt_project.config.unified_config import UnifiedConfigManager, get_unified_config, reset_config


class TestUnifiedConfigManager:
    """Test unified configuration manager."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = UnifiedConfigManager()
        
        assert config.environment in ['development', 'testing']  # Allow testing environment
        assert config.debug == True
        assert config.port == 8000
        assert config.quantum.enabled == True
        assert config.quantum.backend in ['aer_simulator', 'simulator']  # Allow different defaults
        assert config.database.url.startswith('sqlite://')
        assert config.features.enable_fault_tolerance == True  # Use existing feature flag
    
    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            'DEBUG': 'false',
            'PORT': '9000',
            'QUANTUM_BACKEND': 'aer_simulator',
            'QUANTUM_SHOTS': '2048'
        }):
            config = UnifiedConfigManager()
            
            assert config.debug == False
            assert config.port == 9000
            assert config.quantum.backend == 'aer_simulator'
            assert config.quantum.shots == 2048
    
    def test_json_config_loading(self):
        """Test loading from JSON configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "default": {
                    "debug": False,
                    "port": 5000,
                    "quantum": {
                        "backend": "test_backend",
                        "shots": 512
                    }
                },
                "development": {
                    "debug": True,
                    "quantum": {
                        "shots": 1024
                    }
                }
            }
            json.dump(config_data, f)
            f.flush()
            
            with patch.dict(os.environ, {'FLASK_ENV': 'development'}):
                config = UnifiedConfigManager(config_file=f.name, env_file='/nonexistent')
            
                # Should load default config
                assert config.port == 5000
                assert config.quantum.backend == 'test_backend'
                
                # Should override with development config
                assert config.debug == True
                assert config.quantum.shots == 1024
            
            # Cleanup
            os.unlink(f.name)
    
    def test_env_file_loading(self):
        """Test loading from .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("DEBUG=false\n")
            f.write("PORT=7000\n")
            f.write("QUANTUM_BACKEND=custom_backend\n")
            f.write("# This is a comment\n")
            f.write("QUANTUM_SHOTS=4096\n")
            f.flush()
            
            config = UnifiedConfigManager(env_file=f.name, config_file='/nonexistent')
            
            assert config.debug == False
            assert config.port == 7000
            assert config.quantum.backend == 'custom_backend'
            assert config.quantum.shots == 4096
            
            # Cleanup
            os.unlink(f.name)
    
    def test_flask_config(self):
        """Test Flask configuration generation."""
        config = UnifiedConfigManager()
        flask_config = config.get_flask_config()
        
        assert 'DEBUG' in flask_config
        assert 'SECRET_KEY' in flask_config
        assert 'SQLALCHEMY_DATABASE_URI' in flask_config
        assert 'SQLALCHEMY_ENGINE_OPTIONS' in flask_config
    
    def test_quantum_config(self):
        """Test quantum configuration generation."""
        config = UnifiedConfigManager()
        quantum_config = config.get_quantum_config()
        
        assert 'fault_tolerance' in quantum_config
        assert 'holographic_viz' in quantum_config
        assert 'max_qubits' in quantum_config
        assert 'backend' in quantum_config
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid quantum config
        with patch.dict(os.environ, {'QUANTUM_MAX_QUBITS': '0'}):
            with pytest.raises(ValueError, match="max_qubits must be at least 1"):
                UnifiedConfigManager()
        
        with patch.dict(os.environ, {'QUANTUM_SHOTS': '0'}):
            with pytest.raises(ValueError, match="shots must be at least 1"):
                UnifiedConfigManager()
        
        with patch.dict(os.environ, {'QUANTUM_ERROR_THRESHOLD': '2.0'}):
            with pytest.raises(ValueError, match="error_threshold must be between 0 and 1"):
                UnifiedConfigManager()
    
    def test_boolean_env_parsing(self):
        """Test boolean environment variable parsing."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('yes', True),
            ('1', True),
            ('on', True),
            ('false', False),
            ('False', False),
            ('no', False),
            ('0', False),
            ('off', False),
            ('invalid', False)  # Default to False for invalid values
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'DEBUG': env_value}):
                config = UnifiedConfigManager()
                assert config.debug == expected
    
    def test_configuration_summary(self):
        """Test configuration summary generation."""
        config = UnifiedConfigManager()
        summary = config.get_summary()
        
        assert 'environment' in summary
        assert 'quantum_enabled' in summary
        assert 'features_enabled' in summary
        assert 'apis_configured' in summary
        
        # Test structure
        assert isinstance(summary['features_enabled'], dict)
        assert isinstance(summary['apis_configured'], dict)
    
    def test_global_config_instance(self):
        """Test global configuration instance management."""
        # Reset first
        reset_config()
        
        # Get instance
        config1 = get_unified_config()
        config2 = get_unified_config()
        
        # Should be same instance
        assert config1 is config2
        
        # Reset and get new instance
        reset_config()
        config3 = get_unified_config()
        
        # Should be different instance
        assert config1 is not config3
