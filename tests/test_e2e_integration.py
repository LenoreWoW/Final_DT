"""
End-to-End Integration Tests for the Quantum Digital Twin Platform.

Tests the full backend -> engine -> quantum pipeline in 7 scenarios:
1. Conversation -> twin creation flow
2. Simulation with quantum backend
3. All 6 benchmark module results
4. Live benchmark run (classical + quantum)
5. Auth flow (register -> login -> me -> protected endpoint)
6. Data upload flow (CSV -> schema analysis)
7. Query a twin (natural language question)

Plus timing assertions for response latency.

Usage:
    cd /Users/hassanalsahli/Desktop/Final_DT
    venv/bin/python -m pytest tests/test_e2e_integration.py -v
"""

import io
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.database import Base, engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped test client.

    Creates fresh database tables before all tests in this module,
    and drops them afterwards so tests are hermetic.
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def auth_token(client):
    """Register + login and return the JWT token for authenticated requests."""
    unique = uuid.uuid4().hex[:8]
    client.post("/api/auth/register", json={
        "username": f"e2e_user_{unique}",
        "email": f"e2e_{unique}@test.com",
        "password": "securepass123",
    })
    resp = client.post("/api/auth/login", json={
        "username": f"e2e_user_{unique}",
        "password": "securepass123",
    })
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def active_twin(client):
    """
    Create a twin, advance it to ACTIVE, and return its ``id``.

    Also stores the extracted_system via the conversation endpoint
    so that simulate and query have data to work with.
    """
    # Use conversation to create a detailed twin
    r = client.post("/api/conversation/", json={
        "message": (
            "I want to model a hospital with 200 beds, 20 ICU beds, "
            "and 500 patients per day. The goal is to optimize patient "
            "flow and reduce wait times. There are 50 doctors and 100 nurses."
        ),
    })
    assert r.status_code == 200
    twin_id = r.json()["twin_id"]

    # Advance to ACTIVE
    r2 = client.patch(f"/api/twins/{twin_id}", json={"status": "active"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == "active"
    return twin_id


# ===================================================================
# Scenario 1: Full conversation -> twin creation flow
# ===================================================================

class TestConversationTwinCreation:
    """Scenario 1: POST /api/conversation creates a twin and extracts system info."""

    def test_conversation_creates_twin(self, client):
        """Sending a message with no twin_id creates a new twin."""
        start = time.time()
        resp = client.post("/api/conversation/", json={
            "message": (
                "I have a patient named John, age 65, diagnosed with lung cancer "
                "stage III. He is currently on cisplatin chemotherapy. I want to "
                "predict his treatment response and optimize the dosage."
            ),
        })
        elapsed = time.time() - start

        assert resp.status_code == 200
        data = resp.json()

        # Twin was created
        assert "twin_id" in data
        assert data["twin_id"]

        # AI responded
        assert data["message"]

        # Extracted system is present
        assert data.get("extracted_info") is not None
        extracted = data["extracted_info"]
        assert extracted.get("domain") is not None

        # Timing: conversation response < 5 seconds
        assert elapsed < 5.0, f"Conversation took {elapsed:.2f}s (limit 5s)"

    def test_conversation_with_existing_twin(self, client, active_twin):
        """Sending follow-up messages to an existing twin."""
        resp = client.post("/api/conversation/", json={
            "twin_id": active_twin,
            "message": "Add a constraint: maximum 12 hours emergency wait time",
        })
        assert resp.status_code == 200
        assert resp.json()["twin_id"] == active_twin

    def test_conversation_history(self, client, active_twin):
        """GET /api/conversation/{twin_id}/history returns messages."""
        resp = client.get(f"/api/conversation/{active_twin}/history")
        assert resp.status_code == 200
        messages = resp.json()
        assert isinstance(messages, list)
        assert len(messages) >= 1  # At least the creation message


# ===================================================================
# Scenario 2: Simulation with quantum backend
# ===================================================================

class TestSimulationQuantumBackend:
    """Scenario 2: POST /api/twins/{id}/simulate runs quantum simulation."""

    def test_simulation_returns_results(self, client, active_twin):
        start = time.time()
        resp = client.post(f"/api/twins/{active_twin}/simulate", json={
            "time_steps": 50,
            "scenarios": 10,
        })
        elapsed = time.time() - start

        assert resp.status_code == 200
        data = resp.json()

        # Core result fields
        assert data["twin_id"] == active_twin
        assert data["simulation_id"]
        assert data["time_steps"] == 50
        assert data["scenarios_run"] == 10

        # Results include quantum metrics
        results = data["results"]
        assert isinstance(results, dict)
        assert "algorithm" in results or "statistics" in results or "final_state" in results

        # Quantum advantage is reported
        qa = data["quantum_advantage"]
        assert isinstance(qa, dict)

        # Execution time is reported
        assert data["execution_time_seconds"] >= 0

        # Timing: simulation < 10 seconds
        assert elapsed < 10.0, f"Simulation took {elapsed:.2f}s (limit 10s)"

    def test_simulation_fails_on_draft_twin(self, client):
        """Simulation should fail on a DRAFT twin."""
        # Create a draft twin (not promoted to active)
        r = client.post("/api/twins/", json={
            "name": "Draft Twin",
            "description": "This twin stays in draft",
        })
        draft_id = r.json()["id"]

        resp = client.post(f"/api/twins/{draft_id}/simulate", json={
            "time_steps": 10,
            "scenarios": 1,
        })
        assert resp.status_code == 400

    def test_simulation_fails_on_nonexistent_twin(self, client):
        resp = client.post("/api/twins/nonexistent-id/simulate", json={
            "time_steps": 10,
            "scenarios": 1,
        })
        assert resp.status_code == 404


# ===================================================================
# Scenario 3: All 6 benchmark module results
# ===================================================================

HEALTHCARE_MODULES = [
    "personalized_medicine",
    "drug_discovery",
    "medical_imaging",
    "genomic_analysis",
    "epidemic_modeling",
    "hospital_operations",
]


class TestBenchmarkResults:
    """Scenario 3: GET /api/benchmark/results/{module} for all 6 modules."""

    @pytest.mark.parametrize("module_id", HEALTHCARE_MODULES)
    def test_benchmark_result_for_module(self, client, module_id):
        start = time.time()
        resp = client.get(f"/api/benchmark/results/{module_id}")
        elapsed = time.time() - start

        assert resp.status_code == 200
        data = resp.json()

        # Required fields
        assert data["module"] == module_id
        assert data["speedup"] is not None
        assert data["speedup"] > 1.0, f"Expected speedup > 1 for {module_id}"

        # Quantum accuracy > classical accuracy
        assert data["quantum_accuracy"] is not None
        assert data["classical_accuracy"] is not None
        assert data["quantum_accuracy"] > data["classical_accuracy"], (
            f"{module_id}: quantum={data['quantum_accuracy']} "
            f"should be > classical={data['classical_accuracy']}"
        )

        # Timing: benchmark query < 1 second
        assert elapsed < 1.0, f"Benchmark query took {elapsed:.2f}s (limit 1s)"

    def test_all_benchmarks_endpoint(self, client):
        resp = client.get("/api/benchmark/results")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["benchmarks"]) == 6
        assert data["total_quantum_advantage"] > 1.0

    def test_benchmark_modules_list(self, client):
        resp = client.get("/api/benchmark/modules")
        assert resp.status_code == 200
        modules = resp.json()["modules"]
        assert len(modules) == 6
        module_ids = {m["id"] for m in modules}
        assert module_ids == set(HEALTHCARE_MODULES)

    def test_benchmark_methodology(self, client):
        resp = client.get("/api/benchmark/methodology")
        assert resp.status_code == 200
        data = resp.json()
        assert "fairness" in data["sections"]
        assert "reproducibility" in data["sections"]

    def test_benchmark_nonexistent_module(self, client):
        resp = client.get("/api/benchmark/results/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# Scenario 4: Live benchmark run (classical + quantum)
# ===================================================================

class TestLiveBenchmarkRun:
    """Scenario 4: POST /api/benchmark/run/{module_id} runs both engines."""

    def test_live_benchmark_personalized_medicine(self, client):
        resp = client.post("/api/benchmark/run/personalized_medicine", json={
            "module": "personalized_medicine",
            "run_classical": True,
            "run_quantum": True,
            "parameters": {},
        })
        assert resp.status_code == 200
        data = resp.json()

        # Both sides ran
        assert data["classical"] is not None, "Classical result missing"
        assert data["quantum"] is not None, "Quantum result missing"

        # Classical has required keys
        assert "execution_time" in data["classical"]
        assert "accuracy" in data["classical"]
        assert "method" in data["classical"]

        # Quantum has required keys
        assert "execution_time" in data["quantum"]
        assert "accuracy" in data["quantum"]
        assert "method" in data["quantum"]

        # Comparison computed
        assert data["comparison"] is not None
        assert "speedup" in data["comparison"]
        assert "accuracy_improvement" in data["comparison"]

    def test_live_benchmark_quantum_only(self, client):
        resp = client.post("/api/benchmark/run/drug_discovery", json={
            "module": "drug_discovery",
            "run_classical": False,
            "run_quantum": True,
            "parameters": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["classical"] is None
        assert data["quantum"] is not None
        # No comparison when only one side ran
        assert data["comparison"] is None

    def test_live_benchmark_classical_only(self, client):
        resp = client.post("/api/benchmark/run/medical_imaging", json={
            "module": "medical_imaging",
            "run_classical": True,
            "run_quantum": False,
            "parameters": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["classical"] is not None
        assert data["quantum"] is None

    def test_live_benchmark_nonexistent_module(self, client):
        resp = client.post("/api/benchmark/run/nonexistent", json={
            "module": "nonexistent",
            "run_classical": True,
            "run_quantum": True,
            "parameters": {},
        })
        assert resp.status_code == 404


# ===================================================================
# Scenario 5: Auth flow (register -> login -> me -> protected endpoint)
# ===================================================================

class TestAuthFlow:
    """Scenario 5: Full authentication lifecycle."""

    def test_register_new_user(self, client):
        unique = uuid.uuid4().hex[:8]
        start = time.time()
        resp = client.post("/api/auth/register", json={
            "username": f"auth_test_{unique}",
            "email": f"auth_{unique}@example.com",
            "password": "password123",
        })
        elapsed = time.time() - start

        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == f"auth_test_{unique}"
        assert data["email"] == f"auth_{unique}@example.com"
        assert "id" in data

        # Timing: auth endpoint < 2 seconds
        assert elapsed < 2.0, f"Register took {elapsed:.2f}s (limit 2s)"

    def test_login_and_get_token(self, client):
        unique = uuid.uuid4().hex[:8]
        client.post("/api/auth/register", json={
            "username": f"login_test_{unique}",
            "email": f"login_{unique}@example.com",
            "password": "password123",
        })

        start = time.time()
        resp = client.post("/api/auth/login", json={
            "username": f"login_test_{unique}",
            "password": "password123",
        })
        elapsed = time.time() - start

        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["username"] == f"login_test_{unique}"

        # Timing: login < 2 seconds
        assert elapsed < 2.0, f"Login took {elapsed:.2f}s (limit 2s)"

    def test_get_me_with_token(self, client, auth_token):
        start = time.time()
        resp = client.get("/api/auth/me", headers={
            "Authorization": f"Bearer {auth_token}",
        })
        elapsed = time.time() - start

        assert resp.status_code == 200
        data = resp.json()
        assert "username" in data
        assert "email" in data
        assert "id" in data

        # Timing: me endpoint < 2 seconds
        assert elapsed < 2.0, f"Get /me took {elapsed:.2f}s (limit 2s)"

    def test_me_without_token_fails(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_login_wrong_password(self, client):
        unique = uuid.uuid4().hex[:8]
        client.post("/api/auth/register", json={
            "username": f"wrong_pw_{unique}",
            "email": f"wrong_{unique}@example.com",
            "password": "password123",
        })
        resp = client.post("/api/auth/login", json={
            "username": f"wrong_pw_{unique}",
            "password": "WRONGPASSWORD",
        })
        assert resp.status_code == 401

    def test_register_duplicate_username(self, client):
        unique = uuid.uuid4().hex[:8]
        client.post("/api/auth/register", json={
            "username": f"dup_{unique}",
            "email": f"dup1_{unique}@example.com",
            "password": "password123",
        })
        resp = client.post("/api/auth/register", json={
            "username": f"dup_{unique}",
            "email": f"dup2_{unique}@example.com",
            "password": "password123",
        })
        assert resp.status_code == 409


# ===================================================================
# Scenario 6: Data upload flow (CSV -> schema analysis)
# ===================================================================

class TestDataUpload:
    """Scenario 6: POST /api/data/upload with CSV returns schema analysis."""

    def test_upload_healthcare_csv(self, client):
        csv_content = (
            b"patient_id,age,sex,diagnosis,treatment,tumor_grade,outcome\n"
            b"P001,55,M,lung_cancer,cisplatin,G2,remission\n"
            b"P002,42,F,breast_cancer,tamoxifen,G1,stable\n"
            b"P003,68,M,colon_cancer,folfox,G3,progression\n"
            b"P004,51,F,lung_cancer,pembrolizumab,G2,remission\n"
            b"P005,73,M,prostate_cancer,enzalutamide,G1,stable\n"
        )

        resp = client.post(
            "/api/data/upload",
            files={"file": ("patients.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert resp.status_code == 201
        data = resp.json()

        # File metadata
        assert data["filename"] == "patients.csv"
        assert data["file_type"] == "csv"
        assert data["row_count"] == 5
        assert data["column_count"] == 7

        # Columns detected
        col_names = [c["name"] for c in data["columns"]]
        assert "patient_id" in col_names
        assert "age" in col_names
        assert "diagnosis" in col_names

        # Domain detection
        assert data["detected_domain"] == "healthcare"

        # Suggested mappings present
        assert len(data["suggested_mappings"]) == 7

    def test_upload_json_file(self, client):
        import json as json_lib
        json_data = json_lib.dumps([
            {"stock": "AAPL", "price": 150.0, "volume": 1000000},
            {"stock": "GOOGL", "price": 2800.0, "volume": 500000},
        ]).encode()

        resp = client.post(
            "/api/data/upload",
            files={"file": ("stocks.json", io.BytesIO(json_data), "application/json")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["file_type"] == "json"
        assert data["row_count"] == 2

    def test_upload_empty_file(self, client):
        resp = client.post(
            "/api/data/upload",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        assert resp.status_code == 400

    def test_upload_unsupported_format(self, client):
        resp = client.post(
            "/api/data/upload",
            files={"file": ("data.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        assert resp.status_code == 400


# ===================================================================
# Scenario 7: Query a twin (natural language question)
# ===================================================================

class TestTwinQuery:
    """Scenario 7: POST /api/twins/{id}/query with natural language."""

    def test_prediction_query(self, client, active_twin):
        resp = client.post(f"/api/twins/{active_twin}/query", json={
            "query": "What will happen to patient outcomes in 6 months?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["twin_id"] == active_twin
        assert data["query_type"] == "prediction"
        assert data["answer"]
        assert data["confidence"] > 0
        assert data["quantum_metrics"]

    def test_optimization_query(self, client, active_twin):
        resp = client.post(f"/api/twins/{active_twin}/query", json={
            "query": "What is the optimal staffing schedule?",
        })
        assert resp.status_code == 200
        assert resp.json()["query_type"] == "optimization"

    def test_understanding_query(self, client, active_twin):
        resp = client.post(f"/api/twins/{active_twin}/query", json={
            "query": "Why are wait times increasing?",
        })
        assert resp.status_code == 200
        assert resp.json()["query_type"] == "understanding"

    def test_comparison_query(self, client, active_twin):
        resp = client.post(f"/api/twins/{active_twin}/query", json={
            "query": "Compare treatment A versus treatment B",
        })
        assert resp.status_code == 200
        assert resp.json()["query_type"] == "comparison"

    def test_query_nonexistent_twin(self, client):
        resp = client.post("/api/twins/nonexistent/query", json={
            "query": "test",
        })
        assert resp.status_code == 404

    def test_query_with_explicit_type(self, client, active_twin):
        resp = client.post(f"/api/twins/{active_twin}/query", json={
            "query": "Show me all possibilities",
            "query_type": "exploration",
        })
        assert resp.status_code == 200
        assert resp.json()["query_type"] == "exploration"


# ===================================================================
# Additional integration tests: Twin CRUD + WebSocket + Health
# ===================================================================

class TestPlatformHealth:
    """Test platform health and informational endpoints."""

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert data["version"] == "2.0.0"

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_api_info_endpoint(self, client):
        resp = client.get("/api")
        assert resp.status_code == 200
        data = resp.json()
        assert "builder" in data["sections"]
        assert "showcase" in data["sections"]
        assert "auth" in data["sections"]


class TestTwinCRUD:
    """Test full twin CRUD lifecycle."""

    def test_create_list_get_update_delete(self, client):
        # Create
        r = client.post("/api/twins/", json={
            "name": "CRUD Test Twin",
            "description": "Testing full lifecycle",
            "domain": "healthcare",
        })
        assert r.status_code == 201
        twin = r.json()
        twin_id = twin["id"]
        assert twin["status"] == "draft"

        # List
        r = client.get("/api/twins/")
        assert r.status_code == 200
        twins = r.json()
        assert any(t["id"] == twin_id for t in twins)

        # Get
        r = client.get(f"/api/twins/{twin_id}")
        assert r.status_code == 200
        assert r.json()["id"] == twin_id

        # Update
        r = client.patch(f"/api/twins/{twin_id}", json={
            "name": "Updated CRUD Twin",
            "status": "active",
        })
        assert r.status_code == 200
        assert r.json()["name"] == "Updated CRUD Twin"
        assert r.json()["status"] == "active"

        # Delete
        r = client.delete(f"/api/twins/{twin_id}")
        assert r.status_code == 204

        # Confirm deletion
        r = client.get(f"/api/twins/{twin_id}")
        assert r.status_code == 404


class TestWebSocket:
    """Test WebSocket endpoint for real-time updates."""

    def test_websocket_connect_and_ping(self, client):
        with client.websocket_connect("/ws/test-twin-123") as ws:
            # Should receive connection confirmation
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["twin_id"] == "test-twin-123"

            # Send ping
            ws.send_json({"action": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"

    def test_websocket_unknown_action(self, client):
        with client.websocket_connect("/ws/test-twin-456") as ws:
            ws.receive_json()  # consume connection msg
            ws.send_json({"action": "unknown_action"})
            resp = ws.receive_json()
            assert resp["type"] == "ack"
            assert resp["received_action"] == "unknown_action"


# ===================================================================
# Timing summary test (aggregated)
# ===================================================================

class TestResponseTimes:
    """Verify all endpoints meet response-time requirements."""

    def test_conversation_under_5s(self, client):
        start = time.time()
        client.post("/api/conversation/", json={
            "message": "Model a patient with diabetes and hypertension",
        })
        assert time.time() - start < 5.0

    def test_simulation_under_10s(self, client, active_twin):
        start = time.time()
        client.post(f"/api/twins/{active_twin}/simulate", json={
            "time_steps": 100,
            "scenarios": 100,
        })
        assert time.time() - start < 10.0

    def test_benchmark_query_under_1s(self, client):
        start = time.time()
        client.get("/api/benchmark/results/personalized_medicine")
        assert time.time() - start < 1.0

    def test_auth_register_under_2s(self, client):
        unique = uuid.uuid4().hex[:8]
        start = time.time()
        client.post("/api/auth/register", json={
            "username": f"timing_{unique}",
            "email": f"timing_{unique}@test.com",
            "password": "testpass123",
        })
        assert time.time() - start < 2.0

    def test_auth_login_under_2s(self, client):
        unique = uuid.uuid4().hex[:8]
        client.post("/api/auth/register", json={
            "username": f"time_login_{unique}",
            "email": f"time_login_{unique}@test.com",
            "password": "testpass123",
        })
        start = time.time()
        client.post("/api/auth/login", json={
            "username": f"time_login_{unique}",
            "password": "testpass123",
        })
        assert time.time() - start < 2.0


# ===================================================================
# Scenario 8: Healthcare Showcase / Benchmark Features
# ===================================================================

class TestHealthcareShowcase:
    """End-to-end tests for the healthcare showcase / benchmark features."""

    def test_showcase_modules_are_healthcare(self, client):
        """GET /api/benchmark/modules — all 6 modules are healthcare-related."""
        resp = client.get("/api/benchmark/modules")
        assert resp.status_code == 200
        modules = resp.json()["modules"]
        assert len(modules) == 6

        expected_ids = set(HEALTHCARE_MODULES)
        actual_ids = {m["id"] for m in modules}
        assert actual_ids == expected_ids, (
            f"Expected modules {expected_ids}, got {actual_ids}"
        )

        # Each module should have a name and description
        for m in modules:
            assert m.get("name"), f"Module {m['id']} missing name"
            assert m.get("description"), f"Module {m['id']} missing description"

    @pytest.mark.parametrize("module_id", HEALTHCARE_MODULES)
    def test_showcase_quantum_advantage_for_each_module(self, client, module_id):
        """Each module shows quantum accuracy > classical and speedup > 1."""
        resp = client.get(f"/api/benchmark/results/{module_id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["module"] == module_id

        # Quantum accuracy should exceed classical accuracy
        assert data["quantum_accuracy"] > data["classical_accuracy"], (
            f"{module_id}: quantum_accuracy={data['quantum_accuracy']} "
            f"should be > classical_accuracy={data['classical_accuracy']}"
        )

        # Speedup must be greater than 1
        assert data["speedup"] > 1.0, (
            f"{module_id}: speedup={data['speedup']} should be > 1.0"
        )

    def test_showcase_total_quantum_advantage(self, client):
        """GET /api/benchmark/results — total_quantum_advantage > 1."""
        resp = client.get("/api/benchmark/results")
        assert resp.status_code == 200
        data = resp.json()

        assert "total_quantum_advantage" in data
        assert data["total_quantum_advantage"] > 1.0, (
            f"total_quantum_advantage={data['total_quantum_advantage']} should be > 1"
        )

    def test_showcase_live_run_returns_comparison(self, client):
        """POST /api/benchmark/run/personalized_medicine with both engines returns comparison."""
        resp = client.post("/api/benchmark/run/personalized_medicine", json={
            "module": "personalized_medicine",
            "run_classical": True,
            "run_quantum": True,
            "parameters": {},
        })
        assert resp.status_code == 200
        data = resp.json()

        # Both sides must have run
        assert data["classical"] is not None
        assert data["quantum"] is not None

        # Comparison must be present with key metrics
        comparison = data["comparison"]
        assert comparison is not None, "comparison should be present when both engines run"
        assert "speedup" in comparison, "comparison missing 'speedup'"
        assert "accuracy_improvement" in comparison, "comparison missing 'accuracy_improvement'"

    def test_showcase_methodology_mentions_aer(self, client):
        """GET /api/benchmark/methodology — Aer simulator is mentioned."""
        resp = client.get("/api/benchmark/methodology")
        assert resp.status_code == 200
        data = resp.json()

        # Flatten the methodology response to a single string for searching
        import json as json_lib
        methodology_text = json_lib.dumps(data).lower()
        assert "aer" in methodology_text, (
            "Benchmark methodology should mention the Qiskit Aer simulator"
        )
