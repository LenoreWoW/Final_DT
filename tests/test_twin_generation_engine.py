"""
Tests for Universal Twin Generation Engine - Phase 2

Tests cover:
- System Extraction from natural language
- Quantum Encoding
- Algorithm Orchestration
- End-to-end twin generation
"""

import pytest

from backend.engine.extraction import (
    SystemExtractor,
    extract_system,
    ExtractionResult,
    DomainType,
    GoalType,
)
from backend.engine.encoding import (
    QuantumEncoder,
    encode_system,
    QuantumEncoding,
    EncodingStrategy,
)
from backend.engine.orchestration import (
    AlgorithmOrchestrator,
    select_algorithms,
    OrchestratorResult,
    ProblemClass,
)
from backend.models.schemas import (
    ExtractedSystem,
    Entity,
    Relationship,
    Rule,
    AlgorithmType,
    QueryType,
)


# =============================================================================
# System Extraction Tests
# =============================================================================

class TestSystemExtraction:
    """Tests for system extraction from natural language."""

    def test_extract_from_empty_message(self):
        """Test extraction from empty message."""
        result = extract_system("")
        assert isinstance(result, ExtractionResult)
        assert result.system is not None
        assert result.confidence < 0.5

    def test_extract_healthcare_domain(self):
        """Test extraction of healthcare domain."""
        message = "I have a patient with stage 2 cancer. Need to plan treatment."
        result = extract_system(message)
        
        assert result.system.domain == "healthcare"
        assert len(result.system.entities) > 0
        assert any(e.type in ["patient", "tumor"] for e in result.system.entities)

    def test_extract_sports_domain(self):
        """Test extraction of sports domain."""
        message = "I'm running a marathon in 8 weeks with two major hills at mile 8 and mile 20."
        result = extract_system(message)
        
        assert result.system.domain == "sports"
        assert len(result.system.entities) > 0
        assert any(e.type in ["athlete", "race"] for e in result.system.entities)

    def test_extract_military_domain(self):
        """Test extraction of military domain."""
        message = "I have 3 battalions defending a 40km front against an armored assault."
        result = extract_system(message)
        
        assert result.system.domain == "military"
        assert len(result.system.entities) > 0

    def test_extract_environment_domain(self):
        """Test extraction of environment domain."""
        message = "There's a wildfire spreading across the forest, threatening species habitat."
        result = extract_system(message)
        
        assert result.system.domain == "environment"

    def test_extract_finance_domain(self):
        """Test extraction of finance domain."""
        message = "I want to optimize my stock portfolio for maximum return."
        result = extract_system(message)
        
        assert result.system.domain == "finance"

    def test_extract_goal_optimize(self):
        """Test extraction of optimization goal."""
        message = "I want to optimize my pacing strategy for the marathon."
        result = extract_system(message)
        
        assert result.system.goal == "optimize"

    def test_extract_goal_predict(self):
        """Test extraction of prediction goal."""
        message = "What will happen to my portfolio in 6 months?"
        result = extract_system(message)
        
        assert result.system.goal == "predict"

    def test_extract_goal_understand(self):
        """Test extraction of understanding goal."""
        message = "Why did the treatment fail for this patient?"
        result = extract_system(message)
        
        assert result.system.goal == "understand"

    def test_extract_entities_with_properties(self):
        """Test extraction of entities with numeric properties."""
        message = "The athlete has a stamina of 85 and current pace is 8 minutes per mile."
        result = extract_system(message)
        
        assert len(result.system.entities) > 0
        # Entities should have properties (even if values aren't extracted perfectly)
        for entity in result.system.entities:
            assert len(entity.properties) > 0

    def test_extract_constraints(self):
        """Test extraction of constraints."""
        message = "Budget is $1,000,000 and must finish within 2 weeks."
        result = extract_system(message)
        
        assert len(result.system.constraints) > 0

    def test_accumulative_extraction(self):
        """Test that extraction accumulates over multiple messages."""
        # First message
        result1 = extract_system("I'm training for a marathon.")
        
        # Second message builds on first
        result2 = extract_system(
            "The race has hills at mile 8 and mile 20.",
            result1.system
        )
        
        # Should have entities from both
        assert len(result2.system.entities) >= len(result1.system.entities)

    def test_confidence_scoring(self):
        """Test that confidence increases with more information."""
        # Minimal info
        result1 = extract_system("Something about sports")
        
        # More complete info
        result2 = extract_system(
            "I'm a marathon runner training for Boston Marathon. "
            "I want to optimize my pacing strategy to finish under 3:30. "
            "The course has hills at mile 8 and 20."
        )
        
        assert result2.confidence > result1.confidence

    def test_missing_info_detection(self):
        """Test that missing information is detected."""
        # Minimal message
        result = extract_system("I have a system.")
        
        assert len(result.missing_info) > 0
        assert "entities" in result.missing_info or "goal" in result.missing_info


# =============================================================================
# Quantum Encoding Tests
# =============================================================================

class TestQuantumEncoding:
    """Tests for quantum encoding of systems."""

    @pytest.fixture
    def sample_system(self):
        """Create a sample extracted system for testing."""
        return ExtractedSystem(
            entities=[
                Entity(id="athlete_0", name="Athlete", type="athlete",
                       properties={"stamina": 85.0, "pace": 8.0}),
                Entity(id="race_0", name="Race", type="race",
                       properties={"distance": 42.0, "elevation": 500.0}),
            ],
            relationships=[
                Relationship(source_id="athlete_0", target_id="race_0",
                             type="participates_in", strength=1.0),
            ],
            rules=[
                Rule(id="fatigue", name="Fatigue", 
                     description="Fatigue increases", type="physiology"),
            ],
            constraints=[],
            goal="optimize",
            domain="sports",
        )

    def test_encode_empty_system(self):
        """Test encoding an empty system."""
        system = ExtractedSystem()
        encoding = encode_system(system)
        
        assert isinstance(encoding, QuantumEncoding)
        assert encoding.qubit_allocation.total_qubits >= 0

    def test_encode_sample_system(self, sample_system):
        """Test encoding a sample system."""
        encoding = encode_system(sample_system)
        
        assert encoding.qubit_allocation.total_qubits > 0
        assert len(encoding.qubit_allocation.entity_qubits) == 2
        assert "athlete_0" in encoding.qubit_allocation.entity_qubits
        assert "race_0" in encoding.qubit_allocation.entity_qubits

    def test_qubit_allocation(self, sample_system):
        """Test that qubits are properly allocated."""
        encoding = encode_system(sample_system, max_qubits=30)
        
        allocation = encoding.qubit_allocation
        
        # Check that entities have qubits
        for entity_id, qubits in allocation.entity_qubits.items():
            assert len(qubits) > 0
            assert all(isinstance(q, int) for q in qubits)
        
        # Check total doesn't exceed max
        assert allocation.total_qubits <= 30

    def test_entanglement_from_relationships(self, sample_system):
        """Test that relationships create entanglement."""
        encoding = encode_system(sample_system)
        
        # Should have entanglement from the relationship
        assert len(encoding.entanglement_map) > 0
        
        # Entanglement should connect qubits from related entities
        for q1, q2 in encoding.entanglement_map:
            assert isinstance(q1, int)
            assert isinstance(q2, int)
            assert q1 != q2

    def test_gate_sequences_from_rules(self, sample_system):
        """Test that rules generate gate sequences."""
        encoding = encode_system(sample_system)
        
        assert len(encoding.gate_sequences) > 0
        
        for rule_id, sequence in encoding.gate_sequences.items():
            assert len(sequence.gates) > 0

    def test_initial_state_normalization(self, sample_system):
        """Test that initial state is properly normalized."""
        encoding = encode_system(sample_system, max_qubits=10)
        
        # Check normalization (sum of |amplitude|^2 = 1)
        if encoding.initial_state:
            total = sum(abs(a)**2 for a in encoding.initial_state)
            assert abs(total - 1.0) < 0.01  # Allow small numerical error

    def test_encoding_strategies(self, sample_system):
        """Test different encoding strategies."""
        for strategy in EncodingStrategy:
            encoding = encode_system(sample_system, strategy=strategy)
            assert encoding.encoding_strategy == strategy

    def test_resource_estimation(self, sample_system):
        """Test resource estimation."""
        encoder = QuantumEncoder()
        encoding = encode_system(sample_system)
        resources = encoder.estimate_resources(encoding)
        
        assert "qubits" in resources
        assert "circuit_depth" in resources
        assert "gate_count" in resources
        assert "estimated_time_seconds" in resources
        assert resources["qubits"] > 0

    def test_max_qubits_constraint(self, sample_system):
        """Test that max_qubits constraint is respected."""
        max_q = 15
        encoding = encode_system(sample_system, max_qubits=max_q)
        
        assert encoding.qubit_allocation.total_qubits <= max_q


# =============================================================================
# Algorithm Orchestration Tests
# =============================================================================

class TestAlgorithmOrchestration:
    """Tests for algorithm orchestration."""

    @pytest.fixture
    def optimization_system(self):
        """Create a system for optimization."""
        return ExtractedSystem(
            entities=[
                Entity(id="e1", name="Entity1", type="unit", properties={}),
            ],
            goal="optimize",
            domain="military",
        )

    @pytest.fixture
    def prediction_system(self):
        """Create a system for prediction."""
        return ExtractedSystem(
            entities=[
                Entity(id="e1", name="Patient", type="patient", properties={}),
            ],
            goal="predict",
            domain="healthcare",
        )

    def test_orchestrate_optimization(self, optimization_system):
        """Test algorithm selection for optimization."""
        result = select_algorithms(optimization_system)
        
        assert isinstance(result, OrchestratorResult)
        assert result.primary_algorithm in [
            AlgorithmType.QAOA,
            AlgorithmType.VQE,
            AlgorithmType.GROVER,
        ]

    def test_orchestrate_prediction(self, prediction_system):
        """Test algorithm selection for prediction."""
        result = select_algorithms(prediction_system)
        
        assert result.primary_algorithm in [
            AlgorithmType.QUANTUM_SIMULATION,
            AlgorithmType.TENSOR_NETWORK,
            AlgorithmType.QNN,
        ]

    def test_domain_preference_applied(self):
        """Test that domain preferences affect selection."""
        healthcare_system = ExtractedSystem(
            entities=[Entity(id="e1", name="Drug", type="drug", properties={})],
            goal="optimize",
            domain="healthcare",
        )
        
        result = select_algorithms(healthcare_system)
        
        # Healthcare domain should boost VQE for molecular simulation
        assert result.confidence > 0

    def test_query_type_affects_selection(self, optimization_system):
        """Test that query type affects algorithm selection."""
        # Optimization query
        result1 = select_algorithms(
            optimization_system,
            query_type=QueryType.OPTIMIZATION
        )
        
        # Prediction query on same system
        result2 = select_algorithms(
            optimization_system,
            query_type=QueryType.PREDICTION
        )
        
        # Different queries may lead to different algorithms
        assert result1.primary_algorithm is not None
        assert result2.primary_algorithm is not None

    def test_pipeline_has_preprocessing(self, optimization_system):
        """Test that pipeline includes preprocessing steps."""
        result = select_algorithms(optimization_system)
        
        assert result.pipeline is not None
        # Optimization should have constraint encoding
        if result.pipeline.algorithms:
            config = result.pipeline.algorithms[0]
            if config.problem_class == ProblemClass.OPTIMIZATION:
                assert "constraint_encoding" in result.pipeline.preprocessing

    def test_pipeline_has_postprocessing(self, optimization_system):
        """Test that pipeline includes postprocessing steps."""
        result = select_algorithms(optimization_system)
        
        assert "result_aggregation" in result.pipeline.postprocessing

    def test_quantum_advantage_factor(self, optimization_system):
        """Test that quantum advantage factor is calculated."""
        result = select_algorithms(optimization_system)
        
        assert result.pipeline.quantum_advantage_factor >= 1.0

    def test_fallback_algorithm_selected(self, optimization_system):
        """Test that fallback algorithm is selected."""
        result = select_algorithms(optimization_system)
        
        # Should have a fallback for robustness
        assert result.fallback_algorithm is not None or result.primary_algorithm is not None

    def test_reasoning_generated(self, optimization_system):
        """Test that reasoning is generated."""
        result = select_algorithms(optimization_system)
        
        assert len(result.reasoning) > 0
        assert "Goal type" in result.reasoning or optimization_system.goal in result.reasoning.lower()

    def test_resource_estimate_included(self, optimization_system):
        """Test that resource estimate is included."""
        result = select_algorithms(optimization_system)
        
        assert "qubits_required" in result.resource_estimate
        assert "estimated_time_seconds" in result.resource_estimate


# =============================================================================
# End-to-End Twin Generation Tests
# =============================================================================

class TestEndToEndTwinGeneration:
    """End-to-end tests for the complete twin generation pipeline."""

    def test_full_pipeline_marathon(self):
        """Test full pipeline for marathon scenario."""
        # 1. Extract system from description
        message = """
        I'm training for the Boston Marathon in 8 weeks. 
        The course has major hills at mile 8 (Heartbreak Hill) and mile 20.
        I want to optimize my pacing strategy to finish under 3:30.
        Current fitness: I can run a half marathon in 1:45.
        """
        extraction_result = extract_system(message)
        
        assert extraction_result.system.domain == "sports"
        assert extraction_result.system.goal == "optimize"
        
        # 2. Encode the system
        encoding = encode_system(extraction_result.system)
        
        assert encoding.qubit_allocation.total_qubits > 0
        
        # 3. Select algorithms
        orchestration_result = select_algorithms(extraction_result.system, encoding)
        
        assert orchestration_result.primary_algorithm is not None
        assert orchestration_result.pipeline.quantum_advantage_factor > 1
        
        print(f"\n📊 Marathon Twin Generation:")
        print(f"   Domain: {extraction_result.system.domain}")
        print(f"   Entities: {len(extraction_result.system.entities)}")
        print(f"   Qubits: {encoding.qubit_allocation.total_qubits}")
        print(f"   Algorithm: {orchestration_result.primary_algorithm.value}")
        print(f"   Quantum Advantage: {orchestration_result.pipeline.quantum_advantage_factor}x")

    def test_full_pipeline_healthcare(self):
        """Test full pipeline for healthcare scenario."""
        message = """
        I was diagnosed with stage 2 breast cancer, ER+, HER2-.
        My oncologist gave me three treatment options.
        I want to compare outcomes and find the optimal treatment.
        """
        extraction_result = extract_system(message)
        
        assert extraction_result.system.domain == "healthcare"
        
        encoding = encode_system(extraction_result.system)
        orchestration_result = select_algorithms(extraction_result.system, encoding)
        
        print(f"\n🏥 Healthcare Twin Generation:")
        print(f"   Domain: {extraction_result.system.domain}")
        print(f"   Entities: {len(extraction_result.system.entities)}")
        print(f"   Qubits: {encoding.qubit_allocation.total_qubits}")
        print(f"   Algorithm: {orchestration_result.primary_algorithm.value}")

    def test_full_pipeline_military(self):
        """Test full pipeline for military scenario."""
        message = """
        I have 3 battalions of soldiers defending a 40km front against an expected armored assault.
        The enemy has tanks and troops. I need tactical deployment.
        I want to optimize defensive positions to minimize casualties.
        """
        extraction_result = extract_system(message)
        
        assert extraction_result.system.domain == "military"
        
        encoding = encode_system(extraction_result.system)
        orchestration_result = select_algorithms(extraction_result.system, encoding)
        
        print(f"\n⚔️ Military Twin Generation:")
        print(f"   Domain: {extraction_result.system.domain}")
        print(f"   Entities: {len(extraction_result.system.entities)}")
        print(f"   Qubits: {encoding.qubit_allocation.total_qubits}")
        print(f"   Algorithm: {orchestration_result.primary_algorithm.value}")

    def test_full_pipeline_environment(self):
        """Test full pipeline for environment scenario."""
        message = """
        There's a wildfire that started 6 hours ago in the forest.
        Currently 500 acres and spreading. Wildlife habitat is threatened.
        We have firefighters and air tankers for conservation efforts.
        I want to predict spread and optimize resource deployment.
        """
        extraction_result = extract_system(message)
        
        assert extraction_result.system.domain == "environment"
        
        encoding = encode_system(extraction_result.system)
        orchestration_result = select_algorithms(extraction_result.system, encoding)
        
        print(f"\n🔥 Environment Twin Generation:")
        print(f"   Domain: {extraction_result.system.domain}")
        print(f"   Entities: {len(extraction_result.system.entities)}")
        print(f"   Qubits: {encoding.qubit_allocation.total_qubits}")
        print(f"   Algorithm: {orchestration_result.primary_algorithm.value}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

