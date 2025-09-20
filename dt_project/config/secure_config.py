"""
Secure configuration management with enhanced security features.
"""

import os
import secrets
import logging
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from pathlib import Path
import json
import base64

logger = logging.getLogger(__name__)

class SecureConfigManager:
    """
    Enhanced configuration manager with security features:
    - Secure secret key generation
    - Input validation
    - Environment variable validation
    - Secure defaults
    """
    
    def __init__(self):
        self.config = {}
        self._load_secure_config()
        
    def _load_secure_config(self):
        """Load configuration with security validations."""
        # Generate secure secret key if not provided
        secret_key = os.getenv('SECRET_KEY')
        if not secret_key or len(secret_key) < 32:
            logger.warning("No secure SECRET_KEY found, generating random key")
            secret_key = secrets.token_hex(32)
            
        self.config['SECRET_KEY'] = secret_key
        
        # Validate required environment variables
        self._validate_required_env_vars()
        
        # Load other configurations with validation
        self._load_api_config()
        self._load_database_config()
        self._load_security_config()
        
    def _validate_required_env_vars(self):
        """Validate that required environment variables are set."""
        required_vars = {
            'FLASK_ENV': 'development',
            'DATABASE_URL': 'sqlite:///dt_project.db',
        }
        
        for var, default in required_vars.items():
            value = os.getenv(var, default)
            if not value:
                raise ValueError(f"Required environment variable {var} is not set")
            self.config[var] = value
            
    def _load_api_config(self):
        """Load and validate API configuration."""
        self.config['API'] = {
            'WEATHER_ENDPOINT': os.getenv('WEATHER_API_ENDPOINT', 'https://api.open-meteo.com/v1/forecast'),
            'GEOCODING_FORWARD': os.getenv('GEOCODING_FORWARD_ENDPOINT', 'https://geocode.maps.co/search'),
            'GEOCODING_REVERSE': os.getenv('GEOCODING_REVERSE_ENDPOINT', 'https://geocode.maps.co/reverse'),
            'REQUEST_TIMEOUT': int(os.getenv('API_REQUEST_TIMEOUT', '30')),
            'RATE_LIMIT_PER_MINUTE': int(os.getenv('API_RATE_LIMIT', '60')),
        }
        
        # Validate API keys (don't log them)
        api_keys = ['GEOCODING_API_KEY', 'IBMQ_TOKEN']
        for key in api_keys:
            value = os.getenv(key)
            if value and len(value) > 10:  # Basic validation
                self.config['API'][key] = value
            else:
                logger.warning(f"API key {key} not set or invalid")
                
    def _load_database_config(self):
        """Load and validate database configuration."""
        db_url = os.getenv('DATABASE_URL', 'sqlite:///dt_project.db')
        
        # Basic URL validation
        if not db_url.startswith(('sqlite://', 'postgresql://', 'mysql://')):
            raise ValueError(f"Invalid database URL: {db_url}")
            
        self.config['DATABASE'] = {
            'URL': db_url,
            'POOL_SIZE': int(os.getenv('DB_POOL_SIZE', '5')),
            'MAX_OVERFLOW': int(os.getenv('DB_MAX_OVERFLOW', '10')),
            'POOL_TIMEOUT': int(os.getenv('DB_POOL_TIMEOUT', '30')),
        }
        
    def _load_security_config(self):
        """Load security-related configuration."""
        self.config['SECURITY'] = {
            'SESSION_LIFETIME_MINUTES': int(os.getenv('SESSION_LIFETIME', '60')),
            'RATE_LIMIT_ENABLED': os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            'CORS_ORIGINS': os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(','),
            'SECURE_COOKIES': os.getenv('SECURE_COOKIES', 'false').lower() == 'true',
            'CSRF_PROTECTION': os.getenv('CSRF_PROTECTION', 'true').lower() == 'true',
        }
        
        # Rate limiting configuration
        self.config['RATE_LIMITS'] = {
            'DEFAULT': os.getenv('RATE_LIMIT_DEFAULT', '100 per hour'),
            'API_CALLS': os.getenv('RATE_LIMIT_API', '1000 per hour'),
            'QUANTUM_OPERATIONS': os.getenv('RATE_LIMIT_QUANTUM', '10 per hour'),
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value safely."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value (for API keys, tokens, etc.)."""
        value = self.get(key)
        if value:
            # Log access but not the value
            logger.info(f"Accessed secret configuration: {key}")
            return value
        return None
        
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get('FLASK_ENV') == 'production'
        
    def validate_config(self) -> bool:
        """Validate the entire configuration."""
        try:
            # Check secret key
            if len(self.config['SECRET_KEY']) < 32:
                logger.error("SECRET_KEY is too short (minimum 32 characters)")
                return False
                
            # Check database URL
            if not self.config['DATABASE']['URL']:
                logger.error("Database URL is not configured")
                return False
                
            # Check security settings for production
            if self.is_production():
                if not self.config['SECURITY']['SECURE_COOKIES']:
                    logger.warning("SECURE_COOKIES should be enabled in production")
                    
                if self.config.get('FLASK_ENV') != 'production':
                    logger.error("FLASK_ENV should be 'production' in production")
                    return False
                    
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
config = SecureConfigManager()

def get_config() -> SecureConfigManager:
    """Get the global configuration instance."""
    return config

def validate_input(value: str, input_type: str, max_length: int = 1000) -> bool:
    """
    Validate user input to prevent injection attacks.
    
    Args:
        value: The input value to validate
        input_type: Type of input (email, numeric, alphanumeric, etc.)
        max_length: Maximum allowed length
        
    Returns:
        True if input is valid, False otherwise
    """
    if not value or len(value) > max_length:
        return False
        
    if input_type == 'email':
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
        
    elif input_type == 'numeric':
        try:
            float(value)
            return True
        except ValueError:
            return False
            
    elif input_type == 'alphanumeric':
        return value.replace('_', '').replace('-', '').isalnum()
        
    elif input_type == 'coordinate':
        try:
            coord = float(value)
            return -180 <= coord <= 180
        except ValueError:
            return False
            
    return True

def sanitize_input(value: str) -> str:
    """Sanitize input to prevent XSS and injection attacks."""
    if not isinstance(value, str):
        return str(value)
        
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '%', ';', '(', ')', '+', '--']
    for char in dangerous_chars:
        value = value.replace(char, '')
        
    return value.strip()