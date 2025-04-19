import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager that loads settings from config.json and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None, env: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the config.json file. If None, uses default path.
            env: Environment to use (development, testing, production). 
                 If None, reads from FLASK_ENV environment variable or defaults to development.
        """
        # Set default config path if not provided
        if config_path is None:
            self.config_path = Path(os.getenv('CONFIG_PATH', 'config/config.json')).resolve()
        else:
            self.config_path = Path(config_path).resolve()
        
        # Determine environment
        self.env = env or os.getenv('FLASK_ENV', 'development')
        logger.info(f"Initializing configuration for environment: {self.env}")
        
        # Load configuration
        self.config = self._load_config()
        self._override_with_env_vars()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
            # Start with default config and update with environment-specific values
            merged_config = config_data.get('default', {}).copy()
            
            if self.env in config_data:
                self._deep_update(merged_config, config_data[self.env])
                
            return merged_config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using empty config")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error parsing config file at {self.config_path}")
            return {}
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update a dictionary with another dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _override_with_env_vars(self) -> None:
        """Override configuration with environment variables."""
        # Example: WEATHER_API_KEY env var overrides config["api"]["weather"]["api_key"]
        for env_name, env_value in os.environ.items():
            # Convert environment variable names to config paths
            if env_name.startswith(('DT_', 'WEATHER_', 'GEOCODING_', 'QUANTUM_', 'UI_')):
                config_path = self._env_to_config_path(env_name)
                if config_path:
                    self._set_config_value(config_path, env_value)
    
    def _env_to_config_path(self, env_name: str) -> Optional[list]:
        """Convert environment variable name to config path."""
        # Example: WEATHER_API_KEY -> ['api', 'weather', 'api_key']
        if env_name.startswith('DT_'):
            # Remove DT_ prefix
            name = env_name[3:]
        else:
            # Handle special prefixes
            prefixes = {
                'WEATHER_': ['api', 'weather'],
                'GEOCODING_': ['api', 'geocoding'],
                'QUANTUM_': ['quantum'],
                'UI_': ['ui']
            }
            
            for prefix, path_prefix in prefixes.items():
                if env_name.startswith(prefix):
                    parts = env_name[len(prefix):].lower().split('_')
                    return path_prefix + parts
                    
            # Not a recognized prefix
            return None
            
        # Convert snake_case to config path
        return name.lower().split('_')
    
    def _set_config_value(self, path: list, value: str) -> None:
        """Set a value in the config based on a path."""
        current = self.config
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Convert value based on existing type or best guess
        key = path[-1]
        if key in current:
            if isinstance(current[key], bool):
                current[key] = value.lower() in ('true', 'yes', '1', 'on')
            elif isinstance(current[key], int):
                try:
                    current[key] = int(value)
                except ValueError:
                    logger.warning(f"Could not convert {value} to int for {path}")
            elif isinstance(current[key], float):
                try:
                    current[key] = float(value)
                except ValueError:
                    logger.warning(f"Could not convert {value} to float for {path}")
            else:
                current[key] = value
        else:
            # Try to guess type
            if value.lower() in ('true', 'false', 'yes', 'no', 'on', 'off'):
                current[key] = value.lower() in ('true', 'yes', 'on')
            elif value.isdigit():
                current[key] = int(value)
            else:
                try:
                    current[key] = float(value)
                except ValueError:
                    current[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated path to configuration value (e.g., "api.weather.timeout")
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if part not in current:
                return default
            current = current[part]
            
        return current
    
    def get_all(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found")
        return value 