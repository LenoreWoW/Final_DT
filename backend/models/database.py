"""
Database models using SQLAlchemy for persistence.
For MVP, we use SQLite. Production will use PostgreSQL.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Enum, ForeignKey, String, Text, Float, Integer, JSON, Boolean, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .schemas import TwinStatus, AlgorithmType

Base = declarative_base()


# =============================================================================
# Database Models
# =============================================================================

class OrganizationModel(Base):
    """SQLAlchemy model for organizations (SaaS multi-tenancy)."""
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    tier = Column(String(50), default="free")
    max_members = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("UserModel", back_populates="organization")
    twins = relationship("TwinModel", back_populates="organization")


class UserModel(Base):
    """SQLAlchemy model for users."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # SaaS fields
    tier = Column(String(50), default="free")
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=True)
    api_key_hash = Column(String(200), nullable=True)
    rate_limit_override = Column(Integer, nullable=True)
    twins_limit = Column(Integer, default=10)
    queries_per_day_limit = Column(Integer, default=50)
    storage_limit_mb = Column(Integer, default=100)

    organization = relationship("OrganizationModel", back_populates="users")
    api_keys = relationship("ApiKeyModel", back_populates="user")
    usage_records = relationship("UsageTrackingModel", back_populates="user")


class TwinModel(Base):
    """SQLAlchemy model for digital twins."""
    __tablename__ = "twins"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), default=TwinStatus.DRAFT.value)
    domain = Column(String(100), nullable=True)

    # Extracted system as JSON
    extracted_system = Column(JSON, nullable=True)

    # Current state as JSON
    state = Column(JSON, default=dict)

    # Algorithms used (stored as JSON list)
    algorithms_used = Column(JSON, default=list)

    # Quantum metrics
    quantum_metrics = Column(JSON, default=dict)

    # OpenQASM circuits (JSON map of circuit_name -> qasm_string)
    qasm_circuits = Column(JSON, default=dict)

    # SaaS fields
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=True)
    is_public = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    conversations = relationship("ConversationModel", back_populates="twin", cascade="all, delete-orphan")
    simulations = relationship("SimulationModel", back_populates="twin", cascade="all, delete-orphan")
    organization = relationship("OrganizationModel", back_populates="twins")


class ConversationModel(Base):
    """SQLAlchemy model for conversations."""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    twin_id = Column(String(36), ForeignKey("twins.id"), nullable=False)

    # Messages stored as JSON array
    messages = Column(JSON, default=list)

    # Conversation context/metadata (renamed to avoid SQLAlchemy reserved name)
    context = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    twin = relationship("TwinModel", back_populates="conversations")


class SimulationModel(Base):
    """SQLAlchemy model for simulation results."""
    __tablename__ = "simulations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    twin_id = Column(String(36), ForeignKey("twins.id"), nullable=False)

    # Simulation parameters
    time_steps = Column(Integer, default=100)
    scenarios_run = Column(Integer, default=1)
    parameters = Column(JSON, default=dict)

    # Results
    results = Column(JSON, default=dict)
    predictions = Column(JSON, default=list)

    # Quantum advantage metrics
    quantum_advantage = Column(JSON, default=dict)
    execution_time_seconds = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    twin = relationship("TwinModel", back_populates="simulations")


class BenchmarkModel(Base):
    """SQLAlchemy model for benchmark results."""
    __tablename__ = "benchmarks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    module = Column(String(100), nullable=False)

    # Timing
    classical_time_seconds = Column(Float, nullable=True)
    quantum_time_seconds = Column(Float, nullable=True)

    # Accuracy
    classical_accuracy = Column(Float, nullable=True)
    quantum_accuracy = Column(Float, nullable=True)

    # Comparison
    speedup = Column(Float, nullable=True)
    improvement = Column(Float, nullable=True)

    # Details
    details = Column(JSON, default=dict)

    # OpenQASM circuit
    qasm_circuit = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKeyModel(Base):
    """SQLAlchemy model for API keys."""
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(200), nullable=False)
    name = Column(String(100), default="default")
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    user = relationship("UserModel", back_populates="api_keys")


class TierLimitModel(Base):
    """SQLAlchemy model for tier-based limits."""
    __tablename__ = "tier_limits"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tier = Column(String(50), unique=True, nullable=False)
    max_twins = Column(Integer, default=10)
    queries_per_day = Column(Integer, default=50)
    max_qubits = Column(Integer, default=20)
    storage_mb = Column(Integer, default=100)
    concurrent_simulations = Column(Integer, default=1)
    priority = Column(Integer, default=0)  # 0=lowest


class DataUploadModel(Base):
    """SQLAlchemy model for data uploads."""
    __tablename__ = "data_uploads"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True)
    twin_id = Column(String(36), ForeignKey("twins.id"), nullable=True)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    size_bytes = Column(Integer, default=0)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class UsageTrackingModel(Base):
    """SQLAlchemy model for usage tracking."""
    __tablename__ = "usage_tracking"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    endpoint = Column(String(500), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserModel", back_populates="usage_records")


# =============================================================================
# Database Connection
# =============================================================================

def get_database_url() -> str:
    """Get database URL from environment or use SQLite for development."""
    import os
    return os.getenv("DATABASE_URL", "sqlite:///./quantum_twins.db")


def create_db_engine():
    """Create database engine."""
    database_url = get_database_url()
    connect_args = {}

    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(database_url, connect_args=connect_args)


def create_session_factory(engine):
    """Create session factory."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database(engine):
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    _seed_tier_limits(engine)


def _seed_tier_limits(engine):
    """Seed tier limits with defaults if empty."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        if session.query(TierLimitModel).count() == 0:
            defaults = [
                TierLimitModel(tier="free", max_twins=10, queries_per_day=50, max_qubits=20, storage_mb=100, concurrent_simulations=1, priority=0),
                TierLimitModel(tier="pro", max_twins=50, queries_per_day=500, max_qubits=30, storage_mb=1000, concurrent_simulations=3, priority=1),
                TierLimitModel(tier="business", max_twins=200, queries_per_day=5000, max_qubits=40, storage_mb=10000, concurrent_simulations=10, priority=2),
                TierLimitModel(tier="enterprise", max_twins=1000, queries_per_day=100000, max_qubits=50, storage_mb=100000, concurrent_simulations=50, priority=3),
            ]
            session.add_all(defaults)
            session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


# Default engine and session factory
engine = create_db_engine()
SessionLocal = create_session_factory(engine)


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
