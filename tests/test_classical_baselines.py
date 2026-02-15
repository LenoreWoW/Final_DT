"""
Tests for classical baseline modules.

Validates that each classical baseline can be instantiated, produces results
with the expected keys, and that the unified run() dispatcher works correctly.
"""

import time

import pytest

from backend.classical_baselines import run as classical_run
from backend.classical_baselines.personalized_medicine_classical import (
    PersonalizedMedicineClassical,
    PatientProfile,
    run_classical_baseline,
)
from backend.classical_baselines.drug_discovery_classical import (
    DrugDiscoveryClassical,
    run_drug_discovery_classical,
)
from backend.classical_baselines.medical_imaging_classical import (
    MedicalImagingClassical,
    run_medical_imaging_classical,
)
from backend.classical_baselines.genomic_analysis_classical import (
    GenomicAnalysisClassical,
    run_genomic_analysis_classical,
)
from backend.classical_baselines.epidemic_modeling_classical import (
    EpidemicModelingClassical,
    run_epidemic_modeling_classical,
)
from backend.classical_baselines.hospital_operations_classical import (
    HospitalOperationsClassical,
    ResourceType,
    run_hospital_operations_classical,
)


# ---- Module names used across tests ----------------------------------------

ALL_MODULES = [
    "personalized_medicine",
    "drug_discovery",
    "medical_imaging",
    "genomic_analysis",
    "epidemic_modeling",
    "hospital_operations",
]


# =============================================================================
# Instantiation tests
# =============================================================================


class TestInstantiation:
    """Each baseline class can be instantiated with default parameters."""

    def test_personalized_medicine_instantiation(self):
        obj = PersonalizedMedicineClassical()
        assert obj.population_size > 0
        assert obj.generations > 0

    def test_drug_discovery_instantiation(self):
        obj = DrugDiscoveryClassical()
        assert obj.md_simulator is not None
        assert obj.protein_binding_site is not None

    def test_medical_imaging_instantiation(self):
        obj = MedicalImagingClassical(image_size=(64, 64))
        assert obj.cnn is not None
        assert obj.image_size == (64, 64)

    def test_genomic_analysis_instantiation(self):
        obj = GenomicAnalysisClassical(n_components=10, n_estimators=10)
        assert obj.pca is not None
        assert obj.rf is not None

    def test_epidemic_modeling_instantiation(self):
        obj = EpidemicModelingClassical(population_size=50, initial_infected=2)
        assert len(obj.agents) == 50

    def test_hospital_operations_instantiation(self):
        resources = {
            ResourceType.BED: 10,
            ResourceType.DOCTOR: 5,
            ResourceType.NURSE: 8,
        }
        obj = HospitalOperationsClassical(resources=resources)
        assert len(obj.resources) == 3


# =============================================================================
# Direct-call result tests (same calls the benchmark router makes)
# =============================================================================


class TestDirectResults:
    """Each top-level run function produces results with expected keys."""

    def test_personalized_medicine_result_keys(self):
        result = run_classical_baseline({})
        assert "method" in result
        assert "best_treatment" in result
        assert "efficacy" in result["best_treatment"]

    def test_drug_discovery_result_keys(self):
        result = run_drug_discovery_classical(library_size=10)
        assert "method" in result
        assert "screening_time" in result
        assert "best_binding_affinity" in result
        assert isinstance(result["best_binding_affinity"], float)

    def test_medical_imaging_result_keys(self):
        result = run_medical_imaging_classical(num_images=4)
        assert "method" in result
        assert "processing_time" in result
        assert "accuracy" in result
        assert "sensitivity" in result
        assert "specificity" in result

    def test_genomic_analysis_result_keys(self):
        result = run_genomic_analysis_classical(n_genes=50, n_samples=30)
        assert "method" in result
        assert "training_time" in result
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result

    def test_epidemic_modeling_result_keys(self):
        result = run_epidemic_modeling_classical(
            population_size=50, simulation_days=10,
        )
        assert "baseline" in result
        assert "with_intervention" in result
        assert "effectiveness_score" in result
        baseline = result["baseline"]
        assert "simulation_time" in baseline
        assert "method" in baseline

    def test_hospital_operations_result_keys(self):
        result = run_hospital_operations_classical(n_patients=10)
        assert "method" in result
        assert "optimization_time" in result
        assert "patients_scheduled" in result
        assert "average_wait_time" in result


# =============================================================================
# Unified dispatcher tests
# =============================================================================


class TestUnifiedDispatcher:
    """The classical_baselines.run() dispatcher routes to the correct module."""

    @pytest.mark.parametrize("module_id", ALL_MODULES)
    def test_dispatcher_returns_standard_keys(self, module_id):
        """Every module returns execution_time, result, and method."""
        # Use small parameters to keep tests fast
        params = {
            "library_size": 10,
            "num_images": 4,
            "n_genes": 50,
            "n_samples": 30,
            "population_size": 50,
            "simulation_days": 10,
            "n_patients": 10,
        }
        out = classical_run(module_id, params)
        assert "execution_time" in out
        assert "result" in out
        assert "method" in out
        assert isinstance(out["result"], dict)

    def test_unknown_module_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown classical baseline module"):
            classical_run("nonexistent_module")


# =============================================================================
# Timing tests (each module should finish in < 10 s with small inputs)
# =============================================================================


class TestExecutionTime:
    """Each baseline completes within a reasonable time bound."""

    @pytest.mark.parametrize(
        "module_id,params",
        [
            ("personalized_medicine", {}),
            ("drug_discovery", {"library_size": 10}),
            ("medical_imaging", {"num_images": 4}),
            ("genomic_analysis", {"n_genes": 50, "n_samples": 30}),
            ("epidemic_modeling", {"population_size": 50, "simulation_days": 10}),
            ("hospital_operations", {"n_patients": 10}),
        ],
    )
    def test_completes_under_10_seconds(self, module_id, params):
        start = time.time()
        classical_run(module_id, params)
        elapsed = time.time() - start
        assert elapsed < 10, f"{module_id} took {elapsed:.1f}s (limit 10s)"
