"""
User Journey Tests: Benchmark Exploration & Data Upload.

Scenario A – A researcher exploring quantum advantage across all 6 healthcare modules.
Scenario B – A user uploading various data files for twin creation.
"""

import os
from pathlib import Path

import pytest

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "sample_data"

# All six healthcare benchmark module IDs
MODULE_IDS = [
    "personalized_medicine",
    "drug_discovery",
    "medical_imaging",
    "genomic_analysis",
    "epidemic_modeling",
    "hospital_operations",
]


# =============================================================================
# Scenario A – Benchmark Journey
# =============================================================================


class TestBenchmarkJourney:
    """A researcher exploring quantum advantage across all 6 healthcare modules."""

    # ------------------------------------------------------------------
    # 1. List available modules
    # ------------------------------------------------------------------
    def test_list_modules(self, client):
        """GET /api/benchmark/modules returns 6+ modules with expected IDs."""
        resp = client.get("/api/benchmark/modules")
        assert resp.status_code == 200
        data = resp.json()

        assert "modules" in data
        modules = data["modules"]
        assert len(modules) >= 6

        returned_ids = {m["id"] for m in modules}
        for mid in MODULE_IDS:
            assert mid in returned_ids, f"Module '{mid}' missing from /modules response"

        # Every module entry should have standard keys
        for m in modules:
            assert "id" in m
            assert "name" in m
            assert "description" in m
            assert "quantum_speedup" in m

    # ------------------------------------------------------------------
    # 2. View pre-computed results (all modules)
    # ------------------------------------------------------------------
    def test_view_all_results(self, client):
        """GET /api/benchmark/results returns an AllBenchmarksResponse."""
        resp = client.get("/api/benchmark/results")
        assert resp.status_code == 200
        data = resp.json()

        assert "benchmarks" in data
        assert "summary" in data
        assert "total_quantum_advantage" in data

        # Should have a benchmark entry per module
        assert len(data["benchmarks"]) == 6

        # Summary keyed by module id
        for mid in MODULE_IDS:
            assert mid in data["summary"], f"Summary missing for '{mid}'"
            s = data["summary"][mid]
            assert s["quantum_speedup"] > 0
            assert -1.0 <= s["accuracy_improvement"] <= 1.0

        # Overall advantage should be a positive number
        assert data["total_quantum_advantage"] > 0

    # ------------------------------------------------------------------
    # 3-8. Run live benchmark for each of the 6 modules
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("module_id", MODULE_IDS)
    def test_run_benchmark(self, client, module_id):
        """POST /api/benchmark/run/{module_id} returns nested results with classical, quantum, comparison."""
        payload = {
            "parameters": {},
            "run_classical": True,
            "run_quantum": True,
        }
        resp = client.post(f"/api/benchmark/run/{module_id}", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Top-level structure
        assert data["module"] == module_id
        assert "run_id" in data
        assert "classical" in data
        assert "quantum" in data
        assert "comparison" in data

        # Classical sub-result
        cl = data["classical"]
        assert cl is not None, f"Classical result missing for {module_id}"
        assert "method" in cl
        assert "execution_time" in cl
        assert "accuracy" in cl

        # Quantum sub-result
        qt = data["quantum"]
        assert qt is not None, f"Quantum result missing for {module_id}"
        assert "method" in qt
        assert "execution_time" in qt
        assert "accuracy" in qt

        # Comparison
        cmp = data["comparison"]
        assert cmp is not None, f"Comparison missing for {module_id}"
        assert "speedup" in cmp
        assert "accuracy_improvement" in cmp
        assert "quantum_advantage_demonstrated" in cmp

    # ------------------------------------------------------------------
    # 9. View methodology
    # ------------------------------------------------------------------
    def test_view_methodology(self, client):
        """GET /api/benchmark/methodology returns methodology info."""
        resp = client.get("/api/benchmark/methodology")
        assert resp.status_code == 200
        data = resp.json()

        assert data["title"] == "Benchmark Methodology"
        assert "sections" in data
        sections = data["sections"]
        assert "hardware" in sections
        assert "fairness" in sections
        assert "metrics" in sections
        assert "reproducibility" in sections

        # Hardware mentions Qiskit Aer (not IBM Quantum hardware)
        hw = sections["hardware"]
        assert "Aer" in hw.get("quantum_simulator", "")

    # ------------------------------------------------------------------
    # 10. Get specific module result (flat BenchmarkResult)
    # ------------------------------------------------------------------
    def test_get_specific_module_result(self, client):
        """GET /api/benchmark/results/{module_id} returns a flat BenchmarkResult."""
        resp = client.get("/api/benchmark/results/personalized_medicine")
        assert resp.status_code == 200
        data = resp.json()

        assert data["module"] == "personalized_medicine"
        assert isinstance(data["classical_time_seconds"], (int, float))
        assert isinstance(data["quantum_time_seconds"], (int, float))
        assert isinstance(data["speedup"], (int, float))
        assert 0 <= data["classical_accuracy"] <= 1.0
        assert 0 <= data["quantum_accuracy"] <= 1.0
        assert isinstance(data["improvement"], (int, float))
        assert "details" in data
        assert "created_at" in data

    # ------------------------------------------------------------------
    # 11. Nonexistent module returns 404
    # ------------------------------------------------------------------
    def test_nonexistent_module_result_404(self, client):
        """GET /api/benchmark/results/nonexistent returns 404."""
        resp = client.get("/api/benchmark/results/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["detail"].lower()

    # ------------------------------------------------------------------
    # 12. Nonexistent module run returns 404
    # ------------------------------------------------------------------
    def test_nonexistent_module_run_404(self, client):
        """POST /api/benchmark/run/nonexistent returns 404."""
        payload = {"parameters": {}, "run_classical": True, "run_quantum": True}
        resp = client.post("/api/benchmark/run/nonexistent", json=payload)
        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["detail"].lower()


# =============================================================================
# Scenario B – Data Upload Journey
# =============================================================================


class TestDataUploadJourney:
    """A user uploading data files for twin creation."""

    def _read_sample(self, filename: str) -> bytes:
        """Read a sample data file from backend/sample_data/."""
        path = SAMPLE_DATA_DIR / filename
        return path.read_bytes()

    def _assert_upload_response(self, data: dict, filename: str, file_type: str):
        """Common assertions on a successful DataUploadResponse."""
        assert data["filename"] == filename
        assert data["file_type"] == file_type
        assert data["row_count"] > 0
        assert data["column_count"] > 0
        assert "upload_id" in data
        assert "columns" in data
        assert "suggested_mappings" in data
        assert "preview" in data
        assert "uploaded_at" in data
        assert len(data["columns"]) == data["column_count"]
        assert len(data["suggested_mappings"]) == data["column_count"]
        # Preview has at most 5 rows
        assert len(data["preview"]) <= 5

    # ------------------------------------------------------------------
    # 1. Upload patients.csv
    # ------------------------------------------------------------------
    def test_upload_patients_csv(self, client):
        """Upload patients.csv and verify healthcare domain detected."""
        content = self._read_sample("patients.csv")
        resp = client.post(
            "/api/data/upload",
            files={"file": ("patients.csv", content, "text/csv")},
        )
        assert resp.status_code == 201
        data = resp.json()

        self._assert_upload_response(data, "patients.csv", "csv")
        assert data["row_count"] == 10
        assert data["column_count"] == 7

        col_names = [c["name"] for c in data["columns"]]
        assert "patient_id" in col_names
        assert "tumor_type" in col_names
        assert "treatment" in col_names

        # Domain detection should flag healthcare
        assert data["detected_domain"] == "healthcare"

    # ------------------------------------------------------------------
    # 2. Upload training.csv
    # ------------------------------------------------------------------
    def test_upload_training_csv(self, client):
        """Upload training.csv (sports running data)."""
        content = self._read_sample("training.csv")
        resp = client.post(
            "/api/data/upload",
            files={"file": ("training.csv", content, "text/csv")},
        )
        assert resp.status_code == 201
        data = resp.json()

        self._assert_upload_response(data, "training.csv", "csv")
        assert data["row_count"] == 10
        assert data["column_count"] == 8

        col_names = [c["name"] for c in data["columns"]]
        assert "distance_km" in col_names
        assert "avg_hr" in col_names

    # ------------------------------------------------------------------
    # 3. Upload flood.json
    # ------------------------------------------------------------------
    def test_upload_flood_json(self, client):
        """Upload flood.json (environment/flood risk data)."""
        content = self._read_sample("flood.json")
        resp = client.post(
            "/api/data/upload",
            files={"file": ("flood.json", content, "application/json")},
        )
        assert resp.status_code == 201
        data = resp.json()

        self._assert_upload_response(data, "flood.json", "json")
        assert data["row_count"] >= 1
        assert data["column_count"] >= 1

    # ------------------------------------------------------------------
    # 4. Upload logistics.json
    # ------------------------------------------------------------------
    def test_upload_logistics_json(self, client):
        """Upload logistics.json (supply route data)."""
        content = self._read_sample("logistics.json")
        resp = client.post(
            "/api/data/upload",
            files={"file": ("logistics.json", content, "application/json")},
        )
        assert resp.status_code == 201
        data = resp.json()

        self._assert_upload_response(data, "logistics.json", "json")
        assert data["row_count"] >= 1
        assert data["column_count"] >= 1

    # ------------------------------------------------------------------
    # 5. Upload invalid file — rejected
    # ------------------------------------------------------------------
    def test_upload_invalid_file_rejected(self, client):
        """Uploading an unsupported file type returns 400."""
        resp = client.post(
            "/api/data/upload",
            files={"file": ("report.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "unsupported" in data["detail"].lower() or "allowed" in data["detail"].lower()
