"""
Algorithm Orchestrator

Selects and composes quantum algorithms based on:
- Problem type (optimization, simulation, learning, analysis)
- System characteristics
- Available quantum resources
- Performance requirements

This is the "brain" that decides how to process each twin.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.models.schemas import (
    ExtractedSystem,
    AlgorithmType,
    QueryType,
)
from backend.engine.extraction import ExtractionResult, GoalType
from backend.engine.encoding import QuantumEncoding


class ProblemClass(str, Enum):
    """Classification of problem types."""
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    LEARNING = "learning"
    ANALYSIS = "analysis"
    SAMPLING = "sampling"


@dataclass
class AlgorithmConfig:
    """Configuration for a quantum algorithm."""
    algorithm_type: AlgorithmType
    problem_class: ProblemClass
    min_qubits: int = 2
    max_qubits: int = 100
    supports_noise: bool = True
    classical_fallback: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmPipeline:
    """A pipeline of algorithms to execute."""
    algorithms: List[AlgorithmConfig]
    preprocessing: List[str] = field(default_factory=list)
    postprocessing: List[str] = field(default_factory=list)
    estimated_time_seconds: float = 0.0
    quantum_advantage_factor: float = 1.0


@dataclass
class OrchestratorResult:
    """Result of algorithm orchestration."""
    pipeline: AlgorithmPipeline
    primary_algorithm: AlgorithmType
    fallback_algorithm: Optional[AlgorithmType]
    resource_estimate: Dict[str, Any]
    confidence: float
    reasoning: str


class AlgorithmOrchestrator:
    """
    Selects and orchestrates quantum algorithms for digital twins.
    
    Maps user goals and system characteristics to optimal algorithm choices.
    """
    
    # Algorithm selection rules based on goal type
    GOAL_TO_ALGORITHMS: Dict[GoalType, List[Tuple[AlgorithmType, float]]] = {
        GoalType.OPTIMIZE: [
            (AlgorithmType.QAOA, 0.9),
            (AlgorithmType.VQE, 0.7),
            (AlgorithmType.GROVER, 0.5),
        ],
        GoalType.PREDICT: [
            (AlgorithmType.QUANTUM_SIMULATION, 0.9),
            (AlgorithmType.TENSOR_NETWORK, 0.8),
            (AlgorithmType.QNN, 0.6),
        ],
        GoalType.UNDERSTAND: [
            (AlgorithmType.VQC, 0.8),
            (AlgorithmType.TENSOR_NETWORK, 0.7),
            (AlgorithmType.QNN, 0.6),
        ],
        GoalType.EXPLORE: [
            (AlgorithmType.MONTE_CARLO, 0.9),
            (AlgorithmType.QUANTUM_SIMULATION, 0.7),
            (AlgorithmType.TENSOR_NETWORK, 0.6),
        ],
        GoalType.COMPARE: [
            (AlgorithmType.VQC, 0.8),
            (AlgorithmType.QAOA, 0.7),
            (AlgorithmType.QUANTUM_SIMULATION, 0.5),
        ],
        GoalType.DETECT: [
            (AlgorithmType.QNN, 0.9),
            (AlgorithmType.VQC, 0.8),
            (AlgorithmType.GROVER, 0.5),
        ],
    }
    
    # Query type to algorithm mapping
    QUERY_TO_ALGORITHMS: Dict[QueryType, List[AlgorithmType]] = {
        QueryType.PREDICTION: [AlgorithmType.QUANTUM_SIMULATION, AlgorithmType.QNN],
        QueryType.OPTIMIZATION: [AlgorithmType.QAOA, AlgorithmType.VQE],
        QueryType.EXPLORATION: [AlgorithmType.MONTE_CARLO, AlgorithmType.QUANTUM_SIMULATION],
        QueryType.COUNTERFACTUAL: [AlgorithmType.QUANTUM_SIMULATION, AlgorithmType.TENSOR_NETWORK],
        QueryType.UNDERSTANDING: [AlgorithmType.VQC, AlgorithmType.TENSOR_NETWORK],
        QueryType.COMPARISON: [AlgorithmType.QAOA, AlgorithmType.VQC],
    }
    
    # Domain-specific algorithm preferences
    DOMAIN_PREFERENCES: Dict[str, Dict[AlgorithmType, float]] = {
        "healthcare": {
            AlgorithmType.VQE: 1.2,  # Molecular simulation
            AlgorithmType.QNN: 1.1,  # Pattern recognition
            AlgorithmType.TENSOR_NETWORK: 1.2,  # Genomics
        },
        "sports": {
            AlgorithmType.QAOA: 1.2,  # Optimization
            AlgorithmType.QUANTUM_SIMULATION: 1.1,
        },
        "military": {
            AlgorithmType.QAOA: 1.3,  # Strategic optimization
            AlgorithmType.QUANTUM_SIMULATION: 1.2,
        },
        "finance": {
            AlgorithmType.MONTE_CARLO: 1.3,  # Risk analysis
            AlgorithmType.QAOA: 1.2,  # Portfolio optimization
        },
        "environment": {
            AlgorithmType.TENSOR_NETWORK: 1.2,  # Large-scale simulation
            AlgorithmType.QUANTUM_SIMULATION: 1.2,
        },
    }
    
    # Algorithm configurations
    ALGORITHM_CONFIGS: Dict[AlgorithmType, AlgorithmConfig] = {
        AlgorithmType.QAOA: AlgorithmConfig(
            algorithm_type=AlgorithmType.QAOA,
            problem_class=ProblemClass.OPTIMIZATION,
            min_qubits=4,
            max_qubits=50,
            parameters={"layers": 3, "optimizer": "COBYLA"},
        ),
        AlgorithmType.VQE: AlgorithmConfig(
            algorithm_type=AlgorithmType.VQE,
            problem_class=ProblemClass.OPTIMIZATION,
            min_qubits=2,
            max_qubits=30,
            parameters={"ansatz": "UCCSD", "optimizer": "SLSQP"},
        ),
        AlgorithmType.GROVER: AlgorithmConfig(
            algorithm_type=AlgorithmType.GROVER,
            problem_class=ProblemClass.OPTIMIZATION,
            min_qubits=3,
            max_qubits=20,
            supports_noise=False,
        ),
        AlgorithmType.VQC: AlgorithmConfig(
            algorithm_type=AlgorithmType.VQC,
            problem_class=ProblemClass.LEARNING,
            min_qubits=2,
            max_qubits=20,
            parameters={"feature_map": "ZZFeatureMap", "layers": 2},
        ),
        AlgorithmType.QNN: AlgorithmConfig(
            algorithm_type=AlgorithmType.QNN,
            problem_class=ProblemClass.LEARNING,
            min_qubits=4,
            max_qubits=20,
            parameters={"hidden_layers": [8, 4], "activation": "tanh"},
        ),
        AlgorithmType.TENSOR_NETWORK: AlgorithmConfig(
            algorithm_type=AlgorithmType.TENSOR_NETWORK,
            problem_class=ProblemClass.SIMULATION,
            min_qubits=10,
            max_qubits=1000,  # Can handle large systems
            parameters={"bond_dimension": 32},
        ),
        AlgorithmType.QUANTUM_SIMULATION: AlgorithmConfig(
            algorithm_type=AlgorithmType.QUANTUM_SIMULATION,
            problem_class=ProblemClass.SIMULATION,
            min_qubits=4,
            max_qubits=50,
            parameters={"trotter_steps": 10},
        ),
        AlgorithmType.MONTE_CARLO: AlgorithmConfig(
            algorithm_type=AlgorithmType.MONTE_CARLO,
            problem_class=ProblemClass.SAMPLING,
            min_qubits=4,
            max_qubits=30,
            parameters={"samples": 10000},
        ),
    }
    
    def orchestrate(
        self,
        system: ExtractedSystem,
        encoding: Optional[QuantumEncoding] = None,
        query_type: Optional[QueryType] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """
        Select and configure algorithms for a system.
        
        Args:
            system: The extracted system
            encoding: Optional quantum encoding
            query_type: Optional specific query type
            constraints: Optional constraints (max_time, max_qubits, etc.)
            
        Returns:
            OrchestratorResult with algorithm pipeline
        """
        constraints = constraints or {}
        
        # Determine goal type
        goal = GoalType.PREDICT  # default
        if system.goal:
            try:
                goal = GoalType(system.goal)
            except ValueError:
                # Free-text goal that doesn't map to a GoalType enum value;
                # attempt a keyword-based lookup, otherwise keep the default.
                _goal_lower = system.goal.lower()
                for member in GoalType:
                    if member.value in _goal_lower:
                        goal = member
                        break
        
        # Calculate algorithm scores
        scores = self._calculate_algorithm_scores(system, goal, query_type)
        
        # Apply domain preferences
        domain = system.domain or "general"
        scores = self._apply_domain_preferences(scores, domain)
        
        # Apply resource constraints
        if encoding:
            scores = self._apply_resource_constraints(scores, encoding, constraints)
        
        # Select primary and fallback algorithms
        sorted_algorithms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_algorithms[0][0]
        fallback = sorted_algorithms[1][0] if len(sorted_algorithms) > 1 else None
        
        # Build pipeline
        pipeline = self._build_pipeline(primary, fallback, system, encoding)
        
        # Estimate resources
        resource_estimate = self._estimate_resources(pipeline, encoding)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(system, goal, primary, scores)
        
        return OrchestratorResult(
            pipeline=pipeline,
            primary_algorithm=primary,
            fallback_algorithm=fallback,
            resource_estimate=resource_estimate,
            confidence=scores[primary] / max(scores.values()) if max(scores.values()) > 0 else 0.0,
            reasoning=reasoning,
        )
    
    def _calculate_algorithm_scores(
        self,
        system: ExtractedSystem,
        goal: GoalType,
        query_type: Optional[QueryType]
    ) -> Dict[AlgorithmType, float]:
        """Calculate scores for each algorithm."""
        scores = {algo: 0.0 for algo in AlgorithmType}
        
        # Base scores from goal type
        if goal in self.GOAL_TO_ALGORITHMS:
            for algo, score in self.GOAL_TO_ALGORITHMS[goal]:
                scores[algo] += score
        
        # Add scores from query type
        if query_type and query_type in self.QUERY_TO_ALGORITHMS:
            for algo in self.QUERY_TO_ALGORITHMS[query_type]:
                scores[algo] += 0.5
        
        # Boost based on system characteristics
        if len(system.entities) > 10:
            scores[AlgorithmType.TENSOR_NETWORK] += 0.3
        if len(system.relationships) > 5:
            scores[AlgorithmType.QUANTUM_SIMULATION] += 0.2
        if system.constraints:
            scores[AlgorithmType.QAOA] += 0.2
        
        return scores
    
    def _apply_domain_preferences(
        self,
        scores: Dict[AlgorithmType, float],
        domain: str
    ) -> Dict[AlgorithmType, float]:
        """Apply domain-specific preferences."""
        if domain in self.DOMAIN_PREFERENCES:
            prefs = self.DOMAIN_PREFERENCES[domain]
            for algo, multiplier in prefs.items():
                scores[algo] *= multiplier
        return scores
    
    def _apply_resource_constraints(
        self,
        scores: Dict[AlgorithmType, float],
        encoding: QuantumEncoding,
        constraints: Dict[str, Any]
    ) -> Dict[AlgorithmType, float]:
        """Filter algorithms based on resource constraints."""
        n_qubits = encoding.qubit_allocation.total_qubits
        max_qubits = constraints.get("max_qubits", 100)
        
        for algo, config in self.ALGORITHM_CONFIGS.items():
            # Check qubit requirements
            if n_qubits < config.min_qubits or n_qubits > min(config.max_qubits, max_qubits):
                scores[algo] *= 0.1  # Heavily penalize
            
            # Check noise tolerance
            if constraints.get("noisy", True) and not config.supports_noise:
                scores[algo] *= 0.5
        
        return scores
    
    def _build_pipeline(
        self,
        primary: AlgorithmType,
        fallback: Optional[AlgorithmType],
        system: ExtractedSystem,
        encoding: Optional[QuantumEncoding]
    ) -> AlgorithmPipeline:
        """Build the execution pipeline."""
        algorithms = []
        
        # Add primary algorithm
        primary_config = self.ALGORITHM_CONFIGS.get(primary)
        if primary_config:
            algorithms.append(primary_config)
        
        # Add preprocessing based on problem type
        preprocessing = []
        if primary_config and primary_config.problem_class == ProblemClass.LEARNING:
            preprocessing.append("normalize_data")
            preprocessing.append("feature_extraction")
        elif primary_config and primary_config.problem_class == ProblemClass.OPTIMIZATION:
            preprocessing.append("constraint_encoding")
        
        # Add postprocessing
        postprocessing = ["result_aggregation"]
        if len(system.constraints) > 0:
            postprocessing.append("constraint_validation")
        
        # Calculate estimated time
        estimated_time = 1.0  # Base time in seconds
        if encoding:
            estimated_time = encoding.estimated_circuit_depth * 0.01
        
        # Calculate quantum advantage
        quantum_advantage = 1.0
        if primary in [AlgorithmType.QAOA, AlgorithmType.GROVER]:
            quantum_advantage = 100.0  # Quadratic speedup
        elif primary in [AlgorithmType.VQE, AlgorithmType.QUANTUM_SIMULATION]:
            quantum_advantage = 1000.0  # Exponential for chemistry
        elif primary == AlgorithmType.TENSOR_NETWORK:
            quantum_advantage = 10.0  # Polynomial improvement
        
        return AlgorithmPipeline(
            algorithms=algorithms,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            estimated_time_seconds=estimated_time,
            quantum_advantage_factor=quantum_advantage,
        )
    
    def _estimate_resources(
        self,
        pipeline: AlgorithmPipeline,
        encoding: Optional[QuantumEncoding]
    ) -> Dict[str, Any]:
        """Estimate computational resources."""
        if encoding:
            qubits = encoding.qubit_allocation.total_qubits
            depth = encoding.estimated_circuit_depth
        else:
            qubits = 10
            depth = 50
        
        return {
            "qubits_required": qubits,
            "circuit_depth": depth,
            "estimated_shots": 1000,
            "estimated_time_seconds": pipeline.estimated_time_seconds,
            "quantum_advantage_factor": pipeline.quantum_advantage_factor,
            "classical_equivalent_time": (
                pipeline.estimated_time_seconds * pipeline.quantum_advantage_factor
            ),
        }
    
    def _generate_reasoning(
        self,
        system: ExtractedSystem,
        goal: GoalType,
        primary: AlgorithmType,
        scores: Dict[AlgorithmType, float]
    ) -> str:
        """Generate explanation for algorithm selection."""
        reasons = []
        
        reasons.append(f"Goal type: {goal.value} → Suggests {primary.value}")
        
        if system.domain:
            reasons.append(f"Domain: {system.domain}")
        
        if len(system.entities) > 5:
            reasons.append(f"Large system ({len(system.entities)} entities)")
        
        if system.constraints:
            reasons.append(f"Has {len(system.constraints)} constraints")
        
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        reasons.append(f"Top algorithms: {', '.join(a.value for a, _ in top_3)}")
        
        return " | ".join(reasons)
    
    def get_algorithm_for_query(self, query_type: QueryType) -> AlgorithmType:
        """Get the best algorithm for a specific query type."""
        if query_type in self.QUERY_TO_ALGORITHMS:
            return self.QUERY_TO_ALGORITHMS[query_type][0]
        return AlgorithmType.QUANTUM_SIMULATION


# Singleton instance
orchestrator = AlgorithmOrchestrator()


def select_algorithms(
    system: ExtractedSystem,
    encoding: Optional[QuantumEncoding] = None,
    query_type: Optional[QueryType] = None,
) -> OrchestratorResult:
    """Convenience function to select algorithms."""
    return orchestrator.orchestrate(system, encoding, query_type)

