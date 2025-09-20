"""Database models for Quantum Trail application."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class SimulationRun(Base):
    """Model for simulation runs."""
    __tablename__ = 'simulation_runs'
    
    id = Column(Integer, primary_key=True)
    simulation_id = Column(String(50), unique=True, nullable=False, index=True)
    simulation_type = Column(String(50), nullable=False)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Input parameters
    input_params = Column(JSON)
    
    # Results
    results = Column(JSON)
    error_message = Column(Text)
    
    # Performance metrics
    execution_time = Column(Float)
    quantum_advantage = Column(Float)
    
    # User tracking
    user_id = Column(String(100))
    ip_address = Column(String(45))
    
    # Relationships
    quantum_states = relationship("QuantumState", back_populates="simulation")
    measurements = relationship("QuantumMeasurement", back_populates="simulation")
    
    __table_args__ = (
        Index('idx_simulation_type_status', 'simulation_type', 'status'),
        Index('idx_created_at', 'created_at'),
    )

class QuantumState(Base):
    """Model for quantum states."""
    __tablename__ = 'quantum_states'
    
    id = Column(Integer, primary_key=True)
    entity_id = Column(String(100), nullable=False, index=True)
    simulation_id = Column(Integer, ForeignKey('simulation_runs.id'))
    
    # State information
    state_vector = Column(JSON)  # Stored as JSON array
    fidelity = Column(Float, default=1.0)
    entanglement_measure = Column(Float)
    
    # Metadata
    n_qubits = Column(Integer)
    quantum_backend = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Classical correlates
    classical_data = Column(JSON)
    
    # Relationships
    simulation = relationship("SimulationRun", back_populates="quantum_states")

class QuantumMeasurement(Base):
    """Model for quantum measurements."""
    __tablename__ = 'quantum_measurements'
    
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey('simulation_runs.id'))
    entity_id = Column(String(100), nullable=False)
    
    # Measurement data
    measurement_type = Column(String(50))
    measured_value = Column(Float)
    uncertainty = Column(Float)
    basis_state = Column(String(100))
    probability = Column(Float)
    
    # Timestamp
    measured_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    simulation = relationship("SimulationRun", back_populates="measurements")

class AthleteProfile(Base):
    """Model for athlete profiles."""
    __tablename__ = 'athlete_profiles'
    
    id = Column(Integer, primary_key=True)
    athlete_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(200))
    age = Column(Integer)
    sport = Column(String(100))
    
    # Performance metrics
    fitness_level = Column(Float)
    endurance_level = Column(Float)
    strength_level = Column(Float)
    technique_efficiency = Column(Float)
    
    # Training data
    training_hours = Column(Float)
    recovery_time = Column(Float)
    injury_risk = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    performance_predictions = relationship("PerformancePrediction", back_populates="athlete")

class PerformancePrediction(Base):
    """Model for performance predictions."""
    __tablename__ = 'performance_predictions'
    
    id = Column(Integer, primary_key=True)
    athlete_id = Column(Integer, ForeignKey('athlete_profiles.id'))
    prediction_id = Column(String(100), unique=True, nullable=False)
    
    # Prediction data
    prediction_type = Column(String(50))
    predicted_value = Column(Float)
    confidence_level = Column(Float)
    prediction_horizon = Column(Float)  # in days
    
    # Quantum enhancement
    quantum_enhanced = Column(Boolean, default=False)
    quantum_circuit_depth = Column(Integer)
    quantum_advantage_factor = Column(Float)
    
    # Timestamps
    predicted_at = Column(DateTime(timezone=True), server_default=func.now())
    target_date = Column(DateTime(timezone=True))
    
    # Relationships
    athlete = relationship("AthleteProfile", back_populates="performance_predictions")

class MilitaryMission(Base):
    """Model for military missions."""
    __tablename__ = 'military_missions'
    
    id = Column(Integer, primary_key=True)
    mission_id = Column(String(100), unique=True, nullable=False)
    mission_type = Column(String(50))
    status = Column(String(20), default='planning')
    
    # Mission parameters
    location = Column(JSON)  # {lat, lon, terrain}
    weather_conditions = Column(JSON)
    threat_level = Column(Float)
    success_probability = Column(Float)
    
    # Resources
    unit_size = Column(Integer)
    equipment_status = Column(Float)
    supply_level = Column(Float)
    
    # Quantum analysis
    quantum_analyzed = Column(Boolean, default=False)
    optimal_strategy = Column(JSON)
    risk_assessment = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    mission_start = Column(DateTime(timezone=True))
    mission_end = Column(DateTime(timezone=True))

class APIKey(Base):
    """Model for API keys."""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    key_hash = Column(String(256), unique=True, nullable=False)
    user_id = Column(String(100), nullable=False)
    name = Column(String(100))
    
    # Permissions
    permissions = Column(JSON)  # List of allowed endpoints/operations
    rate_limit = Column(Integer, default=1000)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))

class SystemMetrics(Base):
    """Model for system metrics."""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float)
    
    # Context
    component = Column(String(50))  # web, quantum, database, etc.
    tags = Column(JSON)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_metric_type_time', 'metric_type', 'recorded_at'),
    )

class EventLog(Base):
    """Model for event logging."""
    __tablename__ = 'event_logs'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)
    event_name = Column(String(100), nullable=False)
    severity = Column(String(20))  # info, warning, error, critical
    
    # Event details
    message = Column(Text)
    details = Column(JSON)
    user_id = Column(String(100))
    ip_address = Column(String(45))
    
    # Timestamp
    occurred_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_event_type_time', 'event_type', 'occurred_at'),
        Index('idx_severity', 'severity'),
    )