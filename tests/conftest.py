"""
QTwin Test Configuration & Shared Fixtures.
All tests import from backend/ only.
"""
import os
import pytest
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.database import Base, get_db
from backend.engine.extraction.system_extractor import SystemExtractor
from backend.engine.encoding.quantum_encoder import QuantumEncoder
from backend.engine.orchestration.algorithm_orchestrator import AlgorithmOrchestrator
from backend.engine.twin_generator import TwinGenerator
from backend.engine.quantum_modules import registry


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    test_env = {
        'SECRET_KEY': 'test_secret_key_for_testing_only',
        'DATABASE_URL': 'sqlite:///:memory:',
        'DISABLE_QUANTUM': 'true',
        'LOG_LEVEL': 'WARNING',
    }
    with patch.dict(os.environ, test_env):
        yield


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def extractor():
    return SystemExtractor()


@pytest.fixture
def encoder():
    return QuantumEncoder()


@pytest.fixture
def orchestrator():
    return AlgorithmOrchestrator()


@pytest.fixture
def generator():
    return TwinGenerator()


@pytest.fixture
def sample_healthcare_extraction(extractor):
    return extractor.extract("Hospital with 200 beds, 50 doctors, 1000 patients needing scheduling optimization")


@pytest.fixture
def sample_military_extraction(extractor):
    return extractor.extract("Military base with supply depots and transport vehicles under threat conditions")


@pytest.fixture
def sample_sports_extraction(extractor):
    return extractor.extract("Marathon runner with heart rate zones and VO2max data needing training optimization")


@pytest.fixture
def sample_environment_extraction(extractor):
    return extractor.extract("Environmental monitoring station tracking river pollution and flood risk levels")
