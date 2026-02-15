"""
Healthcare Researcher User Journey - Full end-to-end workflow.

Simulates a healthcare researcher who:
1. Creates a hospital digital twin
2. Describes it through conversation (multiple turns)
3. Checks twin state, activates it
4. Runs quantum simulations
5. Queries for predictions and optimizations
6. Downloads QASM circuits
7. Reviews conversation history
8. Cleans up by deleting the twin

All steps are sequential and share state via class-level variables.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.models.database import Base, get_db


# ---------------------------------------------------------------------------
# Class-scoped fixtures so all journey steps share the same DB / client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def journey_db():
    """Class-scoped DB session so all steps in the journey share one database."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    Session = sessionmaker(bind=eng)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="class")
def client(journey_db):
    """Class-scoped TestClient sharing the journey DB."""
    def override_get_db():
        yield journey_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


# ===========================================================================
# Healthcare Researcher User Journey
# ===========================================================================


class TestHealthcareResearcherJourney:
    """
    Sequential user journey for a healthcare researcher.

    ORDERING DEPENDENCY: Tests are numbered (test_step01_ through test_step11_)
    and MUST run in lexicographic order. Each step stores state on
    ``self.__class__`` (twin_id, simulation_id, first_entity_count) that
    subsequent steps depend on. pytest collects tests in definition order by
    default, and the numbering ensures correct ordering even if a plugin
    sorts alphabetically.
    """

    # -- Step 1: Create twin ------------------------------------------------

    def test_step01_create_twin(self, client: TestClient):
        """POST /api/twins/ -- create a hospital digital twin."""
        resp = client.post("/api/twins/", json={
            "name": "City General Hospital",
            "description": (
                "A 200-bed urban hospital with 50 doctors across "
                "emergency, ICU, and general wards. We need to optimize "
                "patient flow and reduce wait times."
            ),
            "domain": "healthcare",
        })
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["name"] == "City General Hospital"
        assert data["status"] == "draft"
        assert data["domain"] == "healthcare"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

        # Save twin_id for subsequent steps
        self.__class__.twin_id = data["id"]

    # -- Step 2: Conversation -- describe the system ------------------------

    def test_step02_conversation_describe(self, client: TestClient):
        """POST /api/conversation/ -- describe ER beds, ICU, wait times."""
        twin_id = self.__class__.twin_id
        resp = client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": (
                "The hospital has 150 general beds and 20 ICU beds with "
                "ventilators. The emergency room handles 300 patients per day "
                "with an average wait time of 45 minutes. We want to optimize "
                "bed allocation to minimize wait times."
            ),
        })
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["twin_id"] == twin_id
        assert "message" in data
        assert "twin_status" in data
        assert "extracted_info" in data

        # Extracted info should have identified some entities
        extracted = data.get("extracted_info")
        if extracted and extracted.get("entities"):
            self.__class__.first_entity_count = len(extracted["entities"])
        else:
            self.__class__.first_entity_count = 0

    # -- Step 3: Conversation -- add staffing details -----------------------

    def test_step03_conversation_add_detail(self, client: TestClient):
        """POST /api/conversation/ -- second turn with staffing details."""
        twin_id = self.__class__.twin_id
        resp = client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": (
                "We have 50 doctors and 100 nurses working in 3 shifts. "
                "The staffing ratio target is 1 nurse per 4 patients. "
                "Peak hours are 8 AM to 2 PM with 60% of daily admissions."
            ),
        })
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["twin_id"] == twin_id
        assert "message" in data
        assert "twin_status" in data

        # Entities should accumulate (or at least not decrease)
        extracted = data.get("extracted_info")
        if extracted and extracted.get("entities"):
            second_entity_count = len(extracted["entities"])
            assert second_entity_count >= self.__class__.first_entity_count, (
                f"Entities should accumulate: {self.__class__.first_entity_count} -> {second_entity_count}"
            )

    # -- Step 4: Check twin updated ----------------------------------------

    def test_step04_check_twin_updated(self, client: TestClient):
        """GET /api/twins/{id} -- verify twin exists and has expected data."""
        twin_id = self.__class__.twin_id
        resp = client.get(f"/api/twins/{twin_id}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["id"] == twin_id
        assert data["name"] == "City General Hospital"
        assert data["domain"] == "healthcare"
        # Twin may have been auto-activated by conversation or still be draft
        assert data["status"] in ("draft", "active")

    # -- Step 5: Activate twin ----------------------------------------------

    def test_step05_activate_twin(self, client: TestClient):
        """PATCH /api/twins/{id} -- set status to active."""
        twin_id = self.__class__.twin_id
        resp = client.patch(f"/api/twins/{twin_id}", json={
            "status": "active",
        })
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["status"] == "active"
        assert data["id"] == twin_id

    # -- Step 6: Run simulation ---------------------------------------------

    def test_step06_run_simulation(self, client: TestClient):
        """POST /api/twins/{id}/simulate -- run quantum simulation."""
        twin_id = self.__class__.twin_id
        resp = client.post(f"/api/twins/{twin_id}/simulate", json={
            "time_steps": 50,
            "scenarios": 10,
            "parameters": {
                "bed_capacity": 200,
                "icu_capacity": 20,
                "daily_admissions": 300,
            },
        })
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["twin_id"] == twin_id
        assert data["time_steps"] == 50
        assert data["scenarios_run"] == 10
        assert "simulation_id" in data
        assert "results" in data
        assert isinstance(data["results"], dict)
        assert "quantum_advantage" in data
        assert isinstance(data["quantum_advantage"], dict)
        assert "execution_time_seconds" in data
        assert data["execution_time_seconds"] >= 0
        assert "created_at" in data

        # Store simulation_id for reference
        self.__class__.simulation_id = data["simulation_id"]

    # -- Step 7: Query prediction -------------------------------------------

    def test_step07_query_prediction(self, client: TestClient):
        """POST /api/twins/{id}/query -- prediction question."""
        twin_id = self.__class__.twin_id
        resp = client.post(f"/api/twins/{twin_id}/query", json={
            "query": "What will happen to patient wait times if we increase daily admissions by 20%?",
        })
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["twin_id"] == twin_id
        assert data["query_type"] == "prediction"
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "quantum_metrics" in data
        assert isinstance(data["quantum_metrics"], dict)

    # -- Step 8: Query optimization -----------------------------------------

    def test_step08_query_optimization(self, client: TestClient):
        """POST /api/twins/{id}/query -- optimization question."""
        twin_id = self.__class__.twin_id
        resp = client.post(f"/api/twins/{twin_id}/query", json={
            "query": "What is the optimal bed allocation to minimize emergency wait times?",
        })
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["twin_id"] == twin_id
        assert data["query_type"] == "optimization"
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    # -- Step 9: Get QASM circuits ------------------------------------------

    def test_step09_get_qasm_circuits(self, client: TestClient):
        """GET /api/twins/{id}/qasm -- verify OPENQASM format."""
        twin_id = self.__class__.twin_id
        resp = client.get(f"/api/twins/{twin_id}/qasm")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["twin_id"] == twin_id
        assert data["twin_name"] == "City General Hospital"
        assert "circuits" in data
        assert isinstance(data["circuits"], dict)
        assert data["circuit_count"] > 0
        assert data["circuit_count"] == len(data["circuits"])

        # Each circuit must contain valid OPENQASM header
        for circuit_name, qasm_str in data["circuits"].items():
            assert isinstance(qasm_str, str), f"Circuit '{circuit_name}' is not a string"
            assert "OPENQASM" in qasm_str, (
                f"Circuit '{circuit_name}' does not contain OPENQASM header"
            )

    # -- Step 10: Get conversation history ----------------------------------

    def test_step10_get_conversation_history(self, client: TestClient):
        """GET /api/conversation/{id}/history -- verify messages recorded."""
        twin_id = self.__class__.twin_id
        resp = client.get(f"/api/conversation/{twin_id}/history")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        messages = resp.json()
        assert isinstance(messages, list)
        # At least one user + assistant pair from conversation turns
        assert len(messages) >= 2, (
            f"Expected at least 2 messages (1 user + 1 assistant), got {len(messages)}"
        )

        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

        # Each message has required fields
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert "timestamp" in msg

    # -- Step 11: Cleanup delete -------------------------------------------

    def test_step11_cleanup_delete(self, client: TestClient):
        """DELETE /api/twins/{id} -- verify 204, then GET returns 404."""
        twin_id = self.__class__.twin_id

        # Delete
        resp = client.delete(f"/api/twins/{twin_id}")
        assert resp.status_code == 204, f"Expected 204, got {resp.status_code}: {resp.text}"

        # Verify twin is gone
        resp = client.get(f"/api/twins/{twin_id}")
        assert resp.status_code == 404, f"Expected 404 after delete, got {resp.status_code}"
