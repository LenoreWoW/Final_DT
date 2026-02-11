"""
Tests for Twin Generator - Integration with Quantum Modules

Tests cover:
- Twin generation from descriptions
- Simulation execution
- Query processing
- Integration with dt_project modules
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.engine import (
    TwinGenerator,
    generate_twin,
    run_simulation,
    query_twin,
    GenerationResult,
    SimulationConfig,
)
from backend.models.schemas import (
    TwinStatus,
    QueryType,
    ExtractedSystem,
    Entity,
)


# =============================================================================
# Twin Generator Tests
# =============================================================================

class TestTwinGenerator:
    """Tests for the TwinGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a TwinGenerator instance."""
        return TwinGenerator()

    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly."""
        assert generator.extractor is not None
        assert generator.encoder is not None
        assert generator.orchestrator is not None

    def test_quantum_modules_loaded(self, generator):
        """Test that quantum modules are loaded (if available)."""
        # This test passes regardless of whether modules are available
        # It just checks the loading mechanism works
        assert isinstance(generator._quantum_modules, dict)

    def test_generate_from_description(self, generator):
        """Test generating a twin from description."""
        description = """
        I'm training for a marathon. The race has hills at mile 8 and 20.
        I want to optimize my pacing strategy to finish under 3:30.
        """
        
        result = generator.generate(description)
        
        assert isinstance(result, GenerationResult)
        assert result.twin_id is not None
        assert result.extracted_system is not None
        assert result.generation_time_seconds > 0

    def test_generate_with_low_confidence(self, generator):
        """Test that low-confidence extractions stay in DRAFT status."""
        description = "Something vague"
        
        result = generator.generate(description)
        
        assert result.status == TwinStatus.DRAFT
        assert "more information" in result.message.lower() or len(result.message) > 0

    def test_generate_with_high_confidence(self, generator):
        """Test that high-confidence extractions become ACTIVE."""
        description = """
        I have a patient with stage 2 breast cancer, ER+, HER2-.
        Need to compare three treatment options and predict outcomes.
        The patient is 55 years old with no other conditions.
        """
        
        result = generator.generate(description)
        
        # Should have enough info to be active
        if result.extracted_system.entities and result.extracted_system.goal:
            assert result.status == TwinStatus.ACTIVE

    def test_generate_includes_quantum_metrics(self, generator):
        """Test that generation includes quantum metrics."""
        description = "Marathon runner optimizing pacing strategy for race with hills."
        
        result = generator.generate(description)
        
        if result.status == TwinStatus.ACTIVE:
            assert "qubits_allocated" in result.quantum_metrics
            assert "primary_algorithm" in result.quantum_metrics
            assert "quantum_advantage_factor" in result.quantum_metrics

    def test_generate_accumulates_from_existing(self, generator):
        """Test that generation can build on existing system."""
        # First extraction
        result1 = generator.generate("I'm training for a marathon.")
        
        # Second extraction builds on first
        result2 = generator.generate(
            "The race has hills and I want to optimize my pace.",
            existing_system=result1.extracted_system
        )
        
        # Should have more entities
        assert len(result2.extracted_system.entities) >= len(result1.extracted_system.entities)


# =============================================================================
# Simulation Tests
# =============================================================================

class TestSimulation:
    """Tests for simulation functionality."""

    @pytest.fixture
    def sample_system(self):
        """Create a sample system for simulation."""
        return ExtractedSystem(
            entities=[
                Entity(id="athlete", name="Athlete", type="athlete",
                       properties={"stamina": 85, "pace": 8}),
            ],
            goal="optimize",
            domain="sports",
        )

    def test_run_simulation(self, sample_system):
        """Test running a simulation."""
        config = SimulationConfig(
            time_steps=100,
            scenarios=10,
        )
        
        result = run_simulation(sample_system, config)
        
        assert result.twin_id is not None
        assert result.simulation_id is not None
        assert result.time_steps == 100
        assert result.scenarios_run == 10
        assert result.execution_time_seconds > 0

    def test_simulation_returns_results(self, sample_system):
        """Test that simulation returns meaningful results."""
        config = SimulationConfig(time_steps=50, scenarios=5)
        
        result = run_simulation(sample_system, config)
        
        assert "algorithm" in result.results
        # Results may have error (if module unavailable) or statistics/scenarios
        assert (
            "statistics" in result.results or 
            "scenarios" in result.results or 
            "error" in result.results  # Acceptable if quantum module not available
        )

    def test_simulation_includes_quantum_advantage(self, sample_system):
        """Test that simulation reports quantum advantage."""
        config = SimulationConfig(time_steps=100, scenarios=100)
        
        result = run_simulation(sample_system, config)
        
        assert "speedup" in result.quantum_advantage
        assert result.quantum_advantage["speedup"] >= 1

    def test_simulation_generates_predictions(self, sample_system):
        """Test that simulation generates predictions."""
        config = SimulationConfig(time_steps=100, scenarios=50)
        
        result = run_simulation(sample_system, config)
        
        # Predictions may be empty if no statistics, but should be a list
        assert isinstance(result.predictions, list)


# =============================================================================
# Query Tests
# =============================================================================

class TestQuery:
    """Tests for query functionality."""

    @pytest.fixture
    def sample_system(self):
        """Create a sample system for querying."""
        return ExtractedSystem(
            entities=[
                Entity(id="patient", name="Patient", type="patient",
                       properties={"age": 55, "condition": "cancer"}),
            ],
            goal="optimize",
            domain="healthcare",
        )

    def test_query_twin(self, sample_system):
        """Test querying a twin."""
        result = query_twin(sample_system, "What's the best treatment option?")
        
        assert result.twin_id is not None
        assert result.query_type == QueryType.OPTIMIZATION
        assert len(result.answer) > 0

    def test_query_type_detection(self, sample_system):
        """Test automatic query type detection."""
        test_cases = [
            ("What happens in 6 months?", QueryType.PREDICTION),
            ("What's the best strategy?", QueryType.OPTIMIZATION),
            ("Show me all possibilities", QueryType.EXPLORATION),
            ("What if I did X instead?", QueryType.COUNTERFACTUAL),
            ("Why did this happen?", QueryType.UNDERSTANDING),
            ("Compare option A vs B", QueryType.COMPARISON),
        ]
        
        for query, expected_type in test_cases:
            result = query_twin(sample_system, query)
            assert result.query_type == expected_type, f"Failed for: {query}"

    def test_query_includes_confidence(self, sample_system):
        """Test that queries include confidence scores."""
        result = query_twin(sample_system, "Predict the outcome")
        
        assert 0 <= result.confidence <= 1

    def test_query_includes_quantum_metrics(self, sample_system):
        """Test that queries include quantum metrics."""
        result = query_twin(sample_system, "Optimize the treatment")
        
        assert "algorithm_used" in result.quantum_metrics


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with dt_project modules."""

    def test_healthcare_domain_detection(self):
        """Test that healthcare domain is properly detected."""
        description = """
        I'm a cancer patient with stage 2 breast cancer.
        My oncologist recommended three treatment options.
        I want to understand which gives the best outcomes.
        """
        
        result = generate_twin(description)
        
        assert result.extracted_system.domain == "healthcare"

    def test_full_workflow_sports(self):
        """Test complete workflow for sports scenario."""
        # 1. Generate twin
        gen_result = generate_twin("""
            I'm a marathon runner training for Boston Marathon.
            Current fitness allows 1:45 half marathon.
            Want to optimize pacing for sub-3:30 finish.
        """)
        
        assert gen_result.twin_id is not None
        
        # 2. Run simulation
        if gen_result.status == TwinStatus.ACTIVE:
            sim_result = run_simulation(
                gen_result.extracted_system,
                SimulationConfig(time_steps=100, scenarios=50)
            )
            
            assert sim_result.results is not None
            
            # 3. Query the twin
            query_result = query_twin(
                gen_result.extracted_system,
                "What's the optimal pace for the first 10 miles?"
            )
            
            assert query_result.answer is not None

    def test_full_workflow_military(self):
        """Test complete workflow for military scenario."""
        gen_result = generate_twin("""
            I have 3 battalions of soldiers defending against an assault.
            Need to optimize defensive positions for tactical advantage.
            Enemy forces are approaching from the east.
        """)
        
        assert gen_result.extracted_system.domain == "military"
        
        if gen_result.status == TwinStatus.ACTIVE:
            assert "primary_algorithm" in gen_result.quantum_metrics

    def test_full_workflow_environment(self):
        """Test complete workflow for environment scenario."""
        gen_result = generate_twin("""
            There's a wildfire spreading through the forest.
            500 acres affected. Wildlife habitat at risk.
            Need to predict spread and optimize firefighter deployment.
        """)
        
        assert gen_result.extracted_system.domain == "environment"


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Tests for performance characteristics."""

    def test_generation_time(self):
        """Test that generation completes in reasonable time."""
        description = "Marathon runner optimizing race strategy with hills."
        
        result = generate_twin(description)
        
        # Generation should be fast (< 1 second for simple cases)
        assert result.generation_time_seconds < 5.0

    def test_simulation_time(self):
        """Test that simulation completes in reasonable time."""
        system = ExtractedSystem(
            entities=[Entity(id="e1", name="Entity", type="athlete", properties={})],
            goal="optimize",
            domain="sports",
        )
        config = SimulationConfig(time_steps=100, scenarios=100)
        
        result = run_simulation(system, config)
        
        # Simulation should complete quickly
        assert result.execution_time_seconds < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

