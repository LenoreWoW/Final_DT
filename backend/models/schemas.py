"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class TwinStatus(str, Enum):
    """Digital twin lifecycle states."""
    DRAFT = "draft"  # User describing system
    GENERATING = "generating"  # Building quantum circuits
    ACTIVE = "active"  # Ready for simulation
    PAUSED = "paused"  # Temporarily inactive
    LEARNING = "learning"  # Updating from new data
    FAILED = "failed"  # Generation failed
    ARCHIVED = "archived"  # No longer in use


class QueryType(str, Enum):
    """Types of queries users can ask their twin."""
    PREDICTION = "prediction"  # "What happens in 6 months?"
    OPTIMIZATION = "optimization"  # "What's the best strategy?"
    EXPLORATION = "exploration"  # "Show me 1000 possible futures"
    COUNTERFACTUAL = "counterfactual"  # "What if I had done X?"
    UNDERSTANDING = "understanding"  # "Why did X happen?"
    COMPARISON = "comparison"  # "Compare strategy A vs B"


class AlgorithmType(str, Enum):
    """Quantum algorithm types for different problem classes."""
    QAOA = "qaoa"  # Optimization
    VQE = "vqe"  # Molecular simulation
    GROVER = "grover"  # Search
    VQC = "vqc"  # Classification
    QNN = "qnn"  # Neural network
    TENSOR_NETWORK = "tensor_network"  # Large-scale simulation
    QUANTUM_SIMULATION = "quantum_simulation"  # Time evolution
    MONTE_CARLO = "monte_carlo"  # Sampling


# =============================================================================
# System Extraction Schemas
# =============================================================================

class Entity(BaseModel):
    """An entity in the system (person, object, resource, etc.)."""
    id: str
    name: str
    type: str  # e.g., "athlete", "soldier", "molecule", "hospital"
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    """A relationship between entities."""
    source_id: str
    target_id: str
    type: str  # e.g., "competes_with", "supplies_to", "interacts_with"
    strength: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class Rule(BaseModel):
    """A rule governing state changes."""
    id: str
    name: str
    description: str
    formula: Optional[str] = None  # e.g., "fatigue = f(pace, distance)"
    type: str  # "physics", "biology", "economics", "custom"


class Constraint(BaseModel):
    """A constraint on the system."""
    id: str
    name: str
    type: str  # "boundary", "budget", "time", "requirement"
    value: Any
    operator: str = "=="  # "==", "<", ">", "<=", ">="


class ExtractedSystem(BaseModel):
    """Complete extracted system from user description."""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    rules: List[Rule] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    goal: Optional[str] = None  # What the user wants to know/optimize
    domain: Optional[str] = None  # Detected domain (healthcare, military, etc.)


# =============================================================================
# Twin Schemas
# =============================================================================

class TwinCreate(BaseModel):
    """Request to create a new digital twin."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    domain: Optional[str] = None  # Auto-detected if not provided
    initial_data: Optional[Dict[str, Any]] = None


class TwinUpdate(BaseModel):
    """Request to update a twin."""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TwinStatus] = None


class TwinState(BaseModel):
    """Current state of a digital twin."""
    entities: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    time_step: int = 0
    metrics: Dict[str, float] = Field(default_factory=dict)


class Twin(BaseModel):
    """Complete digital twin representation."""
    id: str
    name: str
    description: str
    status: TwinStatus
    domain: Optional[str] = None
    extracted_system: Optional[ExtractedSystem] = None
    state: TwinState = Field(default_factory=TwinState)
    algorithms_used: List[AlgorithmType] = Field(default_factory=list)
    quantum_metrics: Dict[str, Any] = Field(default_factory=dict)
    qasm_circuits: Dict[str, str] = Field(default_factory=dict)
    is_public: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Conversation Schemas
# =============================================================================

class Message(BaseModel):
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationRequest(BaseModel):
    """Request to send a message in conversation."""
    twin_id: Optional[str] = None  # None = creating new twin
    message: str
    context: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Response from conversation endpoint."""
    twin_id: str
    message: str
    extracted_info: Optional[ExtractedSystem] = None
    suggestions: List[str] = Field(default_factory=list)
    twin_status: TwinStatus
    requires_more_info: bool = False
    questions: List[str] = Field(default_factory=list)


# =============================================================================
# Simulation Schemas
# =============================================================================

class SimulationRequest(BaseModel):
    """Request to run a simulation."""
    time_steps: int = Field(default=100, ge=1, le=10000)
    scenarios: int = Field(default=1, ge=1, le=1000)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    what_if: Optional[str] = None  # "what if" scenario description


class SimulationResult(BaseModel):
    """Result of a simulation run."""
    twin_id: str
    simulation_id: str
    time_steps: int
    scenarios_run: int
    results: Dict[str, Any]
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    quantum_advantage: Dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: float
    created_at: datetime


# =============================================================================
# Query Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """Request to query a twin."""
    query: str
    query_type: Optional[QueryType] = None  # Auto-detected if not provided
    parameters: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response to a query."""
    twin_id: str
    query: str
    query_type: QueryType
    answer: str
    data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    quantum_metrics: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Benchmark Schemas (for Showcase)
# =============================================================================

class BenchmarkRequest(BaseModel):
    """Request to run a benchmark comparison."""
    module: Optional[str] = None  # Optional; module_id comes from URL path
    parameters: Dict[str, Any] = Field(default_factory=dict)
    run_classical: bool = True
    run_quantum: bool = True


class BenchmarkResult(BaseModel):
    """Result of a benchmark comparison."""
    module: str
    classical_time_seconds: Optional[float] = None
    quantum_time_seconds: Optional[float] = None
    classical_accuracy: Optional[float] = None
    quantum_accuracy: Optional[float] = None
    speedup: Optional[float] = None
    improvement: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    qasm_circuit: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

