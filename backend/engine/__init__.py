# Universal Twin Generation Engine
#
# The core engine that powers the Quantum Digital Twin Platform.
# Converts natural language descriptions into quantum-powered simulations.

from .extraction import (
    SystemExtractor,
    extract_system,
    ExtractionResult,
    DomainType,
    GoalType,
)

from .encoding import (
    QuantumEncoder,
    encode_system,
    QuantumEncoding,
    QubitAllocation,
    GateSequence,
    EncodingStrategy,
    EntanglementStructure,
)

from .orchestration import (
    AlgorithmOrchestrator,
    select_algorithms,
    OrchestratorResult,
    AlgorithmPipeline,
    AlgorithmConfig,
    ProblemClass,
)

from .twin_generator import (
    TwinGenerator,
    generate_twin,
    run_simulation,
    query_twin,
    GenerationResult,
    SimulationConfig,
)

from .quantum_modules import (
    QuantumResult,
    QuantumModuleRegistry,
    registry as quantum_registry,
    run_qaoa_optimization,
    run_quantum_sensing,
    run_tensor_network,
    run_neural_quantum,
    run_pennylane_ml,
    run_personalized_medicine,
    run_drug_discovery,
    run_medical_imaging,
    run_genomic_analysis,
    run_epidemic_modeling,
    run_hospital_operations,
)

__all__ = [
    # Extraction
    "SystemExtractor",
    "extract_system",
    "ExtractionResult",
    "DomainType",
    "GoalType",

    # Encoding
    "QuantumEncoder",
    "encode_system",
    "QuantumEncoding",
    "QubitAllocation",
    "GateSequence",
    "EncodingStrategy",
    "EntanglementStructure",

    # Orchestration
    "AlgorithmOrchestrator",
    "select_algorithms",
    "OrchestratorResult",
    "AlgorithmPipeline",
    "AlgorithmConfig",
    "ProblemClass",

    # Generator
    "TwinGenerator",
    "generate_twin",
    "run_simulation",
    "query_twin",
    "GenerationResult",
    "SimulationConfig",

    # Quantum Module Wrappers
    "QuantumResult",
    "QuantumModuleRegistry",
    "quantum_registry",
    "run_qaoa_optimization",
    "run_quantum_sensing",
    "run_tensor_network",
    "run_neural_quantum",
    "run_pennylane_ml",
    "run_personalized_medicine",
    "run_drug_discovery",
    "run_medical_imaging",
    "run_genomic_analysis",
    "run_epidemic_modeling",
    "run_hospital_operations",
]
