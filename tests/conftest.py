"""
Test configuration and fixtures.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

@pytest.fixture
def mock_config():
    """Mock configuration manager."""
    config = Mock()
    config.get.return_value = 'test_value'
    config.get_secret.return_value = 'test_secret'
    config.is_production.return_value = False
    config.validate_config.return_value = True
    return config

@pytest.fixture
def temp_db():
    """Temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    yield f"sqlite:///{db_path}"
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)

@pytest.fixture
def sample_athlete_profile():
    """Sample athlete profile for testing."""
    return {
        'name': 'Test Athlete',
        'age': 25,
        'weight': 70.0,
        'height': 175.0,
        'fitness_level': 0.8,
        'experience_level': 'intermediate'
    }

@pytest.fixture
def sample_route_data():
    """Sample route data for testing."""
    return [
        {'latitude': 40.7128, 'longitude': -74.0060, 'elevation': 10.0},
        {'latitude': 40.7130, 'longitude': -74.0062, 'elevation': 12.0},
        {'latitude': 40.7132, 'longitude': -74.0064, 'elevation': 8.0}
    ]

@pytest.fixture
def mock_quantum_backend():
    """Mock quantum backend for testing."""
    backend = Mock()
    backend.name.return_value = 'mock_simulator'
    backend.run.return_value = Mock(result=Mock(get_counts=Mock(return_value={'00': 500, '01': 300, '10': 150, '11': 50})))
    return backend

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env_vars = {
        'FLASK_ENV': 'testing',
        'SECRET_KEY': 'test_secret_key_with_enough_length_for_security',
        'DATABASE_URL': 'sqlite:///:memory:',
        'DISABLE_QUANTUM': 'true',  # Disable quantum operations in tests
        'LOG_LEVEL': 'WARNING'  # Reduce log noise in tests
    }
    
    with patch.dict(os.environ, test_env_vars):
        yield