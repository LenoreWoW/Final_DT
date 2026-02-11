"""
Tests for Benchmark API - Quantum Advantage Showcase

Tests cover:
- Benchmark module listing
- Benchmark results retrieval
- Live benchmark execution
- Methodology documentation
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.models.database import Base, get_db


# =============================================================================
# Test Fixtures
# =============================================================================

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as c:
        yield c
    
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


# =============================================================================
# Module Listing Tests
# =============================================================================

class TestModuleListing:
    """Tests for listing benchmark modules."""

    def test_list_modules(self, client):
        """Test listing all benchmark modules."""
        response = client.get("/api/benchmark/modules")
        assert response.status_code == 200
        
        data = response.json()
        assert "modules" in data
        assert len(data["modules"]) == 6  # 6 healthcare modules

    def test_module_has_required_fields(self, client):
        """Test that each module has required fields."""
        response = client.get("/api/benchmark/modules")
        data = response.json()
        
        for module in data["modules"]:
            assert "id" in module
            assert "name" in module
            assert "description" in module
            assert "quantum_speedup" in module

    def test_all_healthcare_modules_present(self, client):
        """Test that all healthcare modules are listed."""
        response = client.get("/api/benchmark/modules")
        data = response.json()
        
        module_ids = [m["id"] for m in data["modules"]]
        
        expected_modules = [
            "personalized_medicine",
            "drug_discovery",
            "medical_imaging",
            "genomic_analysis",
            "epidemic_modeling",
            "hospital_operations",
        ]
        
        for expected in expected_modules:
            assert expected in module_ids


# =============================================================================
# Benchmark Results Tests
# =============================================================================

class TestBenchmarkResults:
    """Tests for benchmark results."""

    def test_get_all_benchmarks(self, client):
        """Test getting all benchmark results."""
        response = client.get("/api/benchmark/results")
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmarks" in data
        assert "summary" in data
        assert "total_quantum_advantage" in data

    def test_all_modules_have_results(self, client):
        """Test that all modules have benchmark results."""
        response = client.get("/api/benchmark/results")
        data = response.json()
        
        modules_with_results = [b["module"] for b in data["benchmarks"]]
        
        expected_modules = [
            "personalized_medicine",
            "drug_discovery",
            "medical_imaging",
            "genomic_analysis",
            "epidemic_modeling",
            "hospital_operations",
        ]
        
        for expected in expected_modules:
            assert expected in modules_with_results

    def test_benchmark_has_required_fields(self, client):
        """Test that each benchmark has required fields."""
        response = client.get("/api/benchmark/results")
        data = response.json()
        
        for benchmark in data["benchmarks"]:
            assert "module" in benchmark
            assert "classical_time_seconds" in benchmark
            assert "quantum_time_seconds" in benchmark
            assert "classical_accuracy" in benchmark
            assert "quantum_accuracy" in benchmark
            assert "speedup" in benchmark
            assert "improvement" in benchmark

    def test_quantum_outperforms_classical(self, client):
        """Test that quantum shows improvement over classical."""
        response = client.get("/api/benchmark/results")
        data = response.json()
        
        quantum_wins = 0
        for benchmark in data["benchmarks"]:
            if benchmark["quantum_accuracy"] > benchmark["classical_accuracy"]:
                quantum_wins += 1
        
        # Quantum should win in at least 5 out of 6 modules
        assert quantum_wins >= 5

    def test_get_specific_benchmark(self, client):
        """Test getting a specific module's benchmark."""
        response = client.get("/api/benchmark/results/personalized_medicine")
        assert response.status_code == 200
        
        data = response.json()
        assert data["module"] == "personalized_medicine"
        assert data["speedup"] == 1000  # 1000x speedup for this module

    def test_get_nonexistent_benchmark(self, client):
        """Test getting a non-existent module."""
        response = client.get("/api/benchmark/results/fake_module")
        assert response.status_code == 404


# =============================================================================
# Live Benchmark Tests
# =============================================================================

class TestLiveBenchmark:
    """Tests for running live benchmarks."""

    def test_run_benchmark_classical_only(self, client):
        """Test running only classical benchmark."""
        response = client.post(
            "/api/benchmark/run/personalized_medicine",
            json={
                "module": "personalized_medicine",
                "run_classical": True,
                "run_quantum": False,
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["classical"] is not None
        assert data["quantum"] is None

    def test_run_benchmark_quantum_only(self, client):
        """Test running only quantum benchmark."""
        response = client.post(
            "/api/benchmark/run/personalized_medicine",
            json={
                "module": "personalized_medicine",
                "run_classical": False,
                "run_quantum": True,
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["classical"] is None
        assert data["quantum"] is not None

    def test_run_benchmark_both(self, client):
        """Test running both classical and quantum benchmarks."""
        response = client.post(
            "/api/benchmark/run/personalized_medicine",
            json={
                "module": "personalized_medicine",
                "run_classical": True,
                "run_quantum": True,
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["classical"] is not None
        assert data["quantum"] is not None
        assert data["comparison"] is not None

    def test_comparison_includes_speedup(self, client):
        """Test that comparison includes speedup calculation."""
        response = client.post(
            "/api/benchmark/run/personalized_medicine",
            json={
                "module": "personalized_medicine",
                "run_classical": True,
                "run_quantum": True,
            }
        )
        
        data = response.json()
        assert "speedup" in data["comparison"]
        assert data["comparison"]["speedup"] > 0

    def test_run_nonexistent_module(self, client):
        """Test running benchmark on non-existent module."""
        response = client.post(
            "/api/benchmark/run/fake_module",
            json={
                "module": "fake_module",
                "run_classical": True,
                "run_quantum": True,
            }
        )
        assert response.status_code == 404


# =============================================================================
# Methodology Tests
# =============================================================================

class TestMethodology:
    """Tests for benchmark methodology documentation."""

    def test_get_methodology(self, client):
        """Test getting methodology documentation."""
        response = client.get("/api/benchmark/methodology")
        assert response.status_code == 200
        
        data = response.json()
        assert "title" in data
        assert "description" in data
        assert "sections" in data

    def test_methodology_has_hardware_info(self, client):
        """Test that methodology includes hardware information."""
        response = client.get("/api/benchmark/methodology")
        data = response.json()
        
        assert "hardware" in data["sections"]
        assert "classical" in data["sections"]["hardware"]
        assert "quantum_simulator" in data["sections"]["hardware"]

    def test_methodology_has_fairness_info(self, client):
        """Test that methodology includes fairness information."""
        response = client.get("/api/benchmark/methodology")
        data = response.json()
        
        assert "fairness" in data["sections"]
        assert len(data["sections"]["fairness"]) > 0

    def test_methodology_has_metrics_info(self, client):
        """Test that methodology includes metrics information."""
        response = client.get("/api/benchmark/methodology")
        data = response.json()
        
        assert "metrics" in data["sections"]
        assert "accuracy" in data["sections"]["metrics"]
        assert "speedup" in data["sections"]["metrics"]


# =============================================================================
# Classical Baseline Tests
# =============================================================================

class TestClassicalBaselines:
    """Tests for classical baseline implementations."""

    def test_personalized_medicine_baseline(self):
        """Test personalized medicine classical baseline."""
        from backend.classical_baselines.personalized_medicine_classical import (
            PersonalizedMedicineClassical,
            PatientProfile,
        )
        
        patient = PatientProfile(
            age=55,
            cancer_type="breast",
            cancer_stage=2,
            biomarkers={"ER+": 0.8, "HER2+": 0.2},
            comorbidities=[],
        )
        
        optimizer = PersonalizedMedicineClassical(
            population_size=20,
            generations=10,
        )
        
        result = optimizer.optimize(patient)
        
        assert result.best_treatment is not None
        assert result.all_treatments_tested > 0
        assert result.execution_time_seconds > 0
        assert 0 <= result.best_treatment.predicted_efficacy <= 1

    def test_grid_search_baseline(self):
        """Test grid search finds optimal."""
        from backend.classical_baselines.personalized_medicine_classical import (
            PersonalizedMedicineClassical,
            PatientProfile,
        )
        
        patient = PatientProfile(
            age=55,
            cancer_type="breast",
            cancer_stage=2,
            biomarkers={"ER+": 0.8, "HER2+": 0.2},
            comorbidities=[],
        )
        
        optimizer = PersonalizedMedicineClassical()
        result = optimizer.grid_search(patient, available_drugs=["Tamoxifen", "Palbociclib"])
        
        assert result.method == "grid_search"
        assert result.best_treatment is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

