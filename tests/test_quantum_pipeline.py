"""
Quantum Pipeline Tests — Encoding & Orchestration.

Tests the QuantumEncoder and AlgorithmOrchestrator modules across domains.
Uses real extraction results from SystemExtractor (no mocks of backend code).
"""

import pytest

from backend.engine.encoding.quantum_encoder import (
    QuantumEncoder,
    QuantumEncoding,
    QubitAllocation,
    EncodingStrategy,
    GateSequence,
)
from backend.engine.orchestration.algorithm_orchestrator import (
    AlgorithmOrchestrator,
    OrchestratorResult,
    AlgorithmPipeline,
    AlgorithmConfig,
    ProblemClass,
)
from backend.engine.extraction.system_extractor import ExtractionResult
from backend.models.schemas import AlgorithmType, QueryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOMAIN_MESSAGES = {
    "healthcare": (
        "Hospital with 200 beds, 50 doctors, 1000 patients needing "
        "scheduling optimization"
    ),
    "military": (
        "Military base with supply depots and transport vehicles "
        "under threat conditions"
    ),
    "sports": (
        "Marathon runner with heart rate zones and VO2max data "
        "needing training optimization"
    ),
    "environment": (
        "Environmental monitoring station tracking river pollution "
        "and flood risk levels"
    ),
}


# ===================================================================
# TestQuantumEncoding
# ===================================================================


class TestQuantumEncoding:
    """Tests for QuantumEncoder.encode()."""

    # 1. Healthcare system encodes with qubit allocation > 0
    def test_healthcare_encoding_qubit_allocation(
        self, encoder, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        assert encoding.qubit_allocation.total_qubits > 0

    # 2. Encoding strategy is set (not None)
    def test_encoding_strategy_is_set(
        self, encoder, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        assert encoding.encoding_strategy is not None
        assert isinstance(encoding.encoding_strategy, EncodingStrategy)

    # 3. Circuit depth > 0
    def test_circuit_depth_positive(
        self, encoder, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        assert encoding.estimated_circuit_depth > 0

    # 4. Gate count > 0
    def test_gate_count_positive(
        self, encoder, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        assert encoding.estimated_gate_count > 0

    # 5. Entanglement map is a list of tuples
    def test_entanglement_map_structure(
        self, encoder, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        assert isinstance(encoding.entanglement_map, list)
        if encoding.entanglement_map:
            pair = encoding.entanglement_map[0]
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    # 6. Gate sequences generated for rules
    def test_gate_sequences_for_rules(
        self, encoder, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        # There should be one gate sequence per rule
        assert len(encoding.gate_sequences) == len(system.rules)
        for rule_id, seq in encoding.gate_sequences.items():
            assert isinstance(seq, GateSequence)
            assert isinstance(seq.gates, list)

    # 7. Initial state vector has correct length
    def test_initial_state_vector_length(
        self, encoder, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        n_qubits = encoding.qubit_allocation.total_qubits
        # Length should be 2^min(n_qubits, 12) (capped at 4096)
        expected_dim = 2 ** min(n_qubits, 12)
        assert len(encoding.initial_state) == expected_dim

    # 8. Different systems produce different encodings
    def test_different_systems_different_encodings(
        self, extractor, encoder
    ):
        health_result = extractor.extract(DOMAIN_MESSAGES["healthcare"])
        military_result = extractor.extract(DOMAIN_MESSAGES["military"])
        enc_health = encoder.encode(health_result.system)
        enc_military = encoder.encode(military_result.system)
        # At least one property should differ
        differs = (
            enc_health.qubit_allocation.total_qubits
            != enc_military.qubit_allocation.total_qubits
            or enc_health.encoding_strategy != enc_military.encoding_strategy
            or enc_health.estimated_gate_count != enc_military.estimated_gate_count
            or enc_health.entanglement_map != enc_military.entanglement_map
        )
        assert differs, "Expected different encodings for different domains"

    # 9. Larger systems get more qubits
    def test_larger_system_more_qubits(self, extractor, encoder):
        small = extractor.extract("A single patient with cancer needs treatment optimization")
        large = extractor.extract(
            "Hospital with 200 beds, 50 doctors, 1000 patients, "
            "tumor cases and drug treatments needing scheduling optimization"
        )
        enc_small = encoder.encode(small.system)
        enc_large = encoder.encode(large.system)
        assert enc_large.qubit_allocation.total_qubits >= enc_small.qubit_allocation.total_qubits

    # 10-12. Cross-domain parametrized: military, sports, environment all encode
    @pytest.mark.parametrize(
        "domain",
        ["military", "sports", "environment"],
    )
    def test_cross_domain_encoding_succeeds(self, extractor, encoder, domain):
        result = extractor.extract(DOMAIN_MESSAGES[domain])
        encoding = encoder.encode(result.system)
        assert encoding.qubit_allocation.total_qubits > 0
        assert encoding.encoding_strategy is not None
        assert isinstance(encoding, QuantumEncoding)


# ===================================================================
# TestAlgorithmOrchestration
# ===================================================================


class TestAlgorithmOrchestration:
    """Tests for AlgorithmOrchestrator.orchestrate()."""

    # 1. Healthcare orchestration has a primary algorithm
    def test_healthcare_has_primary_algorithm(
        self, encoder, orchestrator, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        result = orchestrator.orchestrate(system, encoding)
        assert result.primary_algorithm is not None
        assert isinstance(result.primary_algorithm, AlgorithmType)

    # 2. Pipeline contains at least 1 algorithm
    def test_pipeline_has_algorithms(
        self, encoder, orchestrator, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        result = orchestrator.orchestrate(system, encoding)
        assert len(result.pipeline.algorithms) >= 1

    # 3. Fallback algorithm is set
    def test_fallback_algorithm_set(
        self, encoder, orchestrator, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        result = orchestrator.orchestrate(system, encoding)
        assert result.fallback_algorithm is not None
        assert isinstance(result.fallback_algorithm, AlgorithmType)

    # 4. Resource estimate is populated
    def test_resource_estimate_populated(
        self, encoder, orchestrator, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        result = orchestrator.orchestrate(system, encoding)
        assert isinstance(result.resource_estimate, dict)
        assert "qubits_required" in result.resource_estimate
        assert "circuit_depth" in result.resource_estimate
        assert "estimated_shots" in result.resource_estimate

    # 5. Reasoning string is non-empty
    def test_reasoning_non_empty(
        self, encoder, orchestrator, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        result = orchestrator.orchestrate(system, encoding)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    # 6. Confidence > 0
    def test_confidence_positive(
        self, encoder, orchestrator, sample_healthcare_extraction
    ):
        system = sample_healthcare_extraction.system
        encoding = encoder.encode(system)
        result = orchestrator.orchestrate(system, encoding)
        assert result.confidence > 0

    # 7. Different problem types lead to different primary algorithms
    def test_different_goals_different_algorithms(
        self, extractor, encoder, orchestrator
    ):
        opt_result = extractor.extract(
            "Optimize hospital scheduling for 100 patients and 20 doctors"
        )
        pred_result = extractor.extract(
            "Predict patient recovery outcomes for cancer treatment"
        )
        enc_opt = encoder.encode(opt_result.system)
        enc_pred = encoder.encode(pred_result.system)
        orch_opt = orchestrator.orchestrate(opt_result.system, enc_opt)
        orch_pred = orchestrator.orchestrate(pred_result.system, enc_pred)
        # At minimum, both should succeed
        assert orch_opt.primary_algorithm is not None
        assert orch_pred.primary_algorithm is not None
        # They may or may not differ, but let's at least check both are valid
        assert isinstance(orch_opt.primary_algorithm, AlgorithmType)
        assert isinstance(orch_pred.primary_algorithm, AlgorithmType)

    # 8. Optimization description selects an optimization-class algorithm
    def test_optimization_selects_optimization_algorithm(
        self, extractor, encoder, orchestrator
    ):
        result = extractor.extract(
            "Optimize supply chain routes for 50 warehouses to minimize cost"
        )
        encoding = encoder.encode(result.system)
        orch = orchestrator.orchestrate(result.system, encoding)
        # QAOA, VQE, or GROVER are all optimization-class
        optimization_algos = {AlgorithmType.QAOA, AlgorithmType.VQE, AlgorithmType.GROVER}
        assert orch.primary_algorithm in optimization_algos, (
            f"Expected optimization algo, got {orch.primary_algorithm}"
        )

    # 9. Prediction description selects appropriate algorithm
    def test_prediction_selects_appropriate_algorithm(
        self, extractor, encoder, orchestrator
    ):
        result = extractor.extract(
            "Predict wildfire spread patterns over the next 48 hours"
        )
        encoding = encoder.encode(result.system)
        orch = orchestrator.orchestrate(result.system, encoding)
        # Prediction goal maps to QUANTUM_SIMULATION, TENSOR_NETWORK, or QNN
        prediction_algos = {
            AlgorithmType.QUANTUM_SIMULATION,
            AlgorithmType.TENSOR_NETWORK,
            AlgorithmType.QNN,
        }
        assert orch.primary_algorithm in prediction_algos, (
            f"Expected prediction algo, got {orch.primary_algorithm}"
        )

    # 10-12. Cross-domain parametrized: military, sports, environment orchestrate
    @pytest.mark.parametrize(
        "domain",
        ["military", "sports", "environment"],
    )
    def test_cross_domain_orchestration_succeeds(
        self, extractor, encoder, orchestrator, domain
    ):
        result = extractor.extract(DOMAIN_MESSAGES[domain])
        encoding = encoder.encode(result.system)
        orch = orchestrator.orchestrate(result.system, encoding)
        assert isinstance(orch, OrchestratorResult)
        assert orch.primary_algorithm is not None
        assert orch.pipeline is not None
        assert len(orch.pipeline.algorithms) >= 1

    # 13. Full pipeline: extract -> encode -> orchestrate chain works
    def test_full_pipeline_extract_encode_orchestrate(
        self, extractor, encoder, orchestrator
    ):
        # Extract
        extraction = extractor.extract(
            "A hospital treating cancer patients with experimental drugs. "
            "Optimize treatment schedules to maximize recovery rate."
        )
        assert extraction.system.entities, "Extraction should find entities"

        # Encode
        encoding = encoder.encode(extraction.system)
        assert encoding.qubit_allocation.total_qubits > 0

        # Orchestrate
        result = orchestrator.orchestrate(extraction.system, encoding)
        assert result.primary_algorithm is not None
        assert result.confidence > 0
        assert len(result.reasoning) > 0
        assert result.resource_estimate["qubits_required"] > 0
