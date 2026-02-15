"""
Cross-Domain User Journey Tests for the Quantum Digital Twin Platform.

Tests three complete user journeys across military, sports, and environment
domains. Each journey follows the sequence:
    create -> converse -> activate -> simulate -> query -> QASM

Uses class-scoped fixtures so sequential tests within a journey share
the same database and twin state.

Usage:
    cd /Users/hassanalsahli/Desktop/Final_DT
    venv/bin/python -m pytest tests/test_user_journey_cross_domain.py -v
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.database import Base, get_db


# ---------------------------------------------------------------------------
# Class-scoped fixture: same DB across all tests in a journey class
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def journey_client():
    """
    Class-scoped test client.

    All tests within the same class share a single SQLite database so that
    sequential journey steps (create -> converse -> activate -> simulate ->
    query -> QASM) can build on each other.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    session.close()
    app.dependency_overrides.clear()


# ===========================================================================
# Military Twin Journey
# ===========================================================================


class TestMilitaryTwinJourney:
    """
    Military logistics commander optimizing supply routes.

    Journey: create -> converse -> activate -> simulate -> query -> QASM
    """

    def test_01_create_twin(self, journey_client):
        """Step 1: Create a military logistics twin."""
        resp = journey_client.post("/api/twins/", json={
            "name": "Military Logistics Network",
            "description": (
                "Military logistics network with 5 bases, 10 transport vehicles, "
                "contested supply routes"
            ),
            "domain": "military",
        })
        assert resp.status_code == 201
        data = resp.json()

        assert data["name"] == "Military Logistics Network"
        assert data["status"] == "draft"
        assert "id" in data

        # Store twin_id for subsequent tests
        self.__class__.twin_id = data["id"]

    def test_02_conversation_add_details(self, journey_client):
        """Step 2: Add threat levels, fuel constraints, ammo requirements via conversation."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": (
                "The network has 5 bases: Alpha, Bravo, Charlie, Delta, Echo. "
                "There are 10 transport vehicles with fuel capacity of 500L each. "
                "Routes between bases have threat levels from 1 to 5. "
                "Each base requires weekly ammunition resupply of 2000 rounds. "
                "Fuel constraints limit convoy range to 300km per trip. "
                "The goal is to optimize supply routes while minimizing exposure "
                "to threat zones."
            ),
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["message"]  # AI responded
        assert data["twin_status"] in ["draft", "active"]

        # Extracted info should have some entities
        if data.get("extracted_info"):
            extracted = data["extracted_info"]
            assert extracted.get("domain") is not None

    def test_03_activate_twin(self, journey_client):
        """Step 3: Activate the military twin for simulation."""
        twin_id = self.__class__.twin_id

        resp = journey_client.patch(f"/api/twins/{twin_id}", json={
            "status": "active",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_04_simulate(self, journey_client):
        """Step 4: Run simulation with 50 time steps and 3 scenarios."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post(f"/api/twins/{twin_id}/simulate", json={
            "time_steps": 50,
            "scenarios": 3,
            "parameters": {
                "threat_level": "high",
                "weather": "sandstorm",
            },
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["simulation_id"]
        assert data["time_steps"] == 50
        assert data["scenarios_run"] == 3
        assert isinstance(data["results"], dict)
        assert data["execution_time_seconds"] >= 0
        assert isinstance(data["quantum_advantage"], dict)

        # Store simulation_id for reference
        self.__class__.simulation_id = data["simulation_id"]

    def test_05_query(self, journey_client):
        """Step 5: Query the safest route from Alpha base to Delta base."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post(f"/api/twins/{twin_id}/query", json={
            "query": "What's the safest route from Alpha base to Delta base?",
            "query_type": "optimization",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["query_type"] == "optimization"
        assert data["answer"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["quantum_metrics"], dict)

    def test_06_get_qasm_circuits(self, journey_client):
        """Step 6: Retrieve QASM circuits for the military twin."""
        twin_id = self.__class__.twin_id

        resp = journey_client.get(f"/api/twins/{twin_id}/qasm")
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["twin_name"] == "Military Logistics Network"
        assert data["circuit_count"] >= 1
        assert isinstance(data["circuits"], dict)

        # Each circuit should be a valid QASM string
        for circuit_name, qasm_str in data["circuits"].items():
            assert isinstance(qasm_str, str)
            assert len(qasm_str) > 0
            assert "OPENQASM" in qasm_str or "qreg" in qasm_str


# ===========================================================================
# Sports Twin Journey
# ===========================================================================


class TestSportsTwinJourney:
    """
    Coach optimizing marathon runner's training.

    Journey: create -> converse -> activate -> simulate -> query -> QASM
    """

    def test_01_create_twin(self, journey_client):
        """Step 1: Create an elite marathon runner twin."""
        resp = journey_client.post("/api/twins/", json={
            "name": "Elite Marathon Runner",
            "description": (
                "Elite marathon runner with HR zones, VO2max 65, "
                "targeting sub-3hr marathon"
            ),
            "domain": "sports",
        })
        assert resp.status_code == 201
        data = resp.json()

        assert data["name"] == "Elite Marathon Runner"
        assert data["status"] == "draft"
        assert "id" in data

        self.__class__.twin_id = data["id"]

    def test_02_conversation_add_details(self, journey_client):
        """Step 2: Add training history, injury history, race schedule."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": (
                "The runner has 5 years of competitive experience. "
                "Training history includes 120km/week peak volume. "
                "Previous injuries: left Achilles tendinitis (2024), "
                "right IT band syndrome (2023). Current VO2max is 65 ml/kg/min. "
                "Heart rate zones: Z1 120-140, Z2 140-155, Z3 155-165, "
                "Z4 165-175, Z5 175-190. Race schedule: half marathon in 4 weeks, "
                "full marathon in 12 weeks. The goal is to optimize training "
                "to achieve sub-3hr marathon while avoiding injury."
            ),
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["message"]
        assert data["twin_status"] in ["draft", "active"]

    def test_03_activate_twin(self, journey_client):
        """Step 3: Activate the sports twin."""
        twin_id = self.__class__.twin_id

        resp = journey_client.patch(f"/api/twins/{twin_id}", json={
            "status": "active",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_04_simulate_training_block(self, journey_client):
        """Step 4: Simulate a training block over 84 time steps (12 weeks)."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post(f"/api/twins/{twin_id}/simulate", json={
            "time_steps": 84,
            "scenarios": 5,
            "parameters": {
                "training_phase": "build",
                "weekly_volume_km": 120,
            },
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["simulation_id"]
        assert data["time_steps"] == 84
        assert data["scenarios_run"] == 5
        assert isinstance(data["results"], dict)
        assert data["execution_time_seconds"] >= 0
        assert isinstance(data["quantum_advantage"], dict)

        self.__class__.simulation_id = data["simulation_id"]

    def test_05_query_race_pace(self, journey_client):
        """Step 5: Query optimal pace at km 30 given expected temperature."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post(f"/api/twins/{twin_id}/query", json={
            "query": (
                "What pace should I hold at km 30 given expected "
                "temperature of 25C?"
            ),
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["query_type"] in [
            "prediction", "optimization", "understanding",
        ]
        assert data["answer"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["quantum_metrics"], dict)

    def test_06_get_qasm_circuits(self, journey_client):
        """Step 6: Retrieve QASM circuits for the sports twin."""
        twin_id = self.__class__.twin_id

        resp = journey_client.get(f"/api/twins/{twin_id}/qasm")
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["twin_name"] == "Elite Marathon Runner"
        assert data["circuit_count"] >= 1
        assert isinstance(data["circuits"], dict)

        for circuit_name, qasm_str in data["circuits"].items():
            assert isinstance(qasm_str, str)
            assert len(qasm_str) > 0
            assert "OPENQASM" in qasm_str or "qreg" in qasm_str


# ===========================================================================
# Environment Twin Journey
# ===========================================================================


class TestEnvironmentTwinJourney:
    """
    Environmental scientist modeling flood risk.

    Journey: create -> converse -> activate -> simulate -> query -> QASM
    """

    def test_01_create_twin(self, journey_client):
        """Step 1: Create a river basin flood monitoring twin."""
        resp = journey_client.post("/api/twins/", json={
            "name": "River Basin Flood Monitor",
            "description": (
                "River basin flood monitoring system with 10 sensor stations"
            ),
            "domain": "environment",
        })
        assert resp.status_code == 201
        data = resp.json()

        assert data["name"] == "River Basin Flood Monitor"
        assert data["status"] == "draft"
        assert "id" in data

        self.__class__.twin_id = data["id"]

    def test_02_conversation_add_details(self, journey_client):
        """Step 2: Add rainfall data, elevation, drainage capacity."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": (
                "The river basin covers 500 sq km with 10 monitoring stations. "
                "Average annual rainfall is 1200mm. Peak drainage capacity is "
                "200 cubic meters per second. Elevation ranges from 50m to 300m. "
                "There are 3 flood-prone zones: Zone A (residential, elevation 55m), "
                "Zone B (industrial, elevation 60m), Zone C (agricultural, elevation 65m). "
                "Historical flood data shows major events every 10 years. "
                "The goal is to predict flood risk and identify areas at highest risk "
                "when rainfall exceeds critical thresholds."
            ),
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["message"]
        assert data["twin_status"] in ["draft", "active"]

    def test_03_activate_twin(self, journey_client):
        """Step 3: Activate the environment twin."""
        twin_id = self.__class__.twin_id

        resp = journey_client.patch(f"/api/twins/{twin_id}", json={
            "status": "active",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_04_simulate_flood_scenario(self, journey_client):
        """Step 4: Simulate a flood scenario with 100 time steps."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post(f"/api/twins/{twin_id}/simulate", json={
            "time_steps": 100,
            "scenarios": 5,
            "parameters": {
                "rainfall_mm_hr": 150,
                "duration_hours": 48,
                "soil_saturation": 0.8,
            },
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["simulation_id"]
        assert data["time_steps"] == 100
        assert data["scenarios_run"] == 5
        assert isinstance(data["results"], dict)
        assert data["execution_time_seconds"] >= 0
        assert isinstance(data["quantum_advantage"], dict)

        self.__class__.simulation_id = data["simulation_id"]

    def test_05_query_flood_risk(self, journey_client):
        """Step 5: Query areas at highest risk if rainfall exceeds 150mm/hr."""
        twin_id = self.__class__.twin_id

        resp = journey_client.post(f"/api/twins/{twin_id}/query", json={
            "query": (
                "What areas are at highest risk if rainfall exceeds 150mm/hr?"
            ),
            "query_type": "prediction",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["query_type"] == "prediction"
        assert data["answer"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["quantum_metrics"], dict)

    def test_06_get_qasm_circuits(self, journey_client):
        """Step 6: Retrieve QASM circuits for the environment twin."""
        twin_id = self.__class__.twin_id

        resp = journey_client.get(f"/api/twins/{twin_id}/qasm")
        assert resp.status_code == 200
        data = resp.json()

        assert data["twin_id"] == twin_id
        assert data["twin_name"] == "River Basin Flood Monitor"
        assert data["circuit_count"] >= 1
        assert isinstance(data["circuits"], dict)

        for circuit_name, qasm_str in data["circuits"].items():
            assert isinstance(qasm_str, str)
            assert len(qasm_str) > 0
            assert "OPENQASM" in qasm_str or "qreg" in qasm_str
