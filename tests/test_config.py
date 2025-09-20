"""
Test configuration management.
"""

import pytest
import tempfile
import json
import os

from dt_project.config.config_manager import ConfigManager


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def test_config_initialization(self):
        """Test config manager initialization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "default": {
                    "app_name": "Test App",
                    "version": "1.0.0",
                    "quantum": {
                        "enabled": True,
                        "backend": "simulator"
                    }
                }
            }
            json.dump(config_data, f)
            f.flush()
            
            config = ConfigManager(config_path=f.name, env='default')
            
            assert config.get('app_name') == "Test App"
            assert config.get('version') == "1.0.0"
            assert config.get('quantum.enabled') == True
            assert config.get('quantum.backend') == "simulator"
            
            # Cleanup
            os.unlink(f.name)
    
    def test_config_defaults(self):
        """Test config defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"default": {"test_key": "test_value"}}
            json.dump(config_data, f)
            f.flush()
            
            config = ConfigManager(config_path=f.name)
            
            assert config.get('test_key') == "test_value"
            assert config.get('nonexistent_key', 'default_val') == 'default_val'
            
            # Cleanup
            os.unlink(f.name)
    
    def test_environment_override(self):
        """Test environment-specific config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "default": {
                    "debug": False,
                    "port": 5000
                },
                "development": {
                    "debug": True,
                    "port": 8000
                }
            }
            json.dump(config_data, f)
            f.flush()
            
            config = ConfigManager(config_path=f.name, env='development')
            
            assert config.get('debug') == True  # Overridden
            assert config.get('port') == 8000  # Overridden
            
            # Cleanup
            os.unlink(f.name)
    
    def test_missing_config_file(self):
        """Test handling missing config file."""
        config = ConfigManager(config_path='/nonexistent/path.json')
        
        # Should not raise error, just return empty config
        assert config.get('any_key', 'default') == 'default'
    
    def test_get_all_config(self):
        """Test getting complete configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "default": {
                    "key1": "value1",
                    "key2": "value2"
                }
            }
            json.dump(config_data, f)
            f.flush()
            
            config = ConfigManager(config_path=f.name)
            all_config = config.get_all()
            
            assert isinstance(all_config, dict)
            assert 'key1' in all_config
            assert 'key2' in all_config
            
            # Cleanup
            os.unlink(f.name)
