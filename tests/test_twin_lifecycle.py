"""
Twin Lifecycle Tests - End-to-end twin generation and full API surface.

Tests cover:
- TwinGenerator engine across multiple domains
- Full REST API lifecycle (CRUD, conversation, history)
- Cross-domain parametrized generation
- WebSocket connectivity
"""

import time
import pytest
from starlette.testclient import TestClient

from backend.engine.twin_generator import TwinGenerator, GenerationResult, SimulationConfig
from backend.models.schemas import TwinStatus, ExtractedSystem


# ---------------------------------------------------------------------------
# Domain descriptions used across tests
# ---------------------------------------------------------------------------

HEALTHCARE_DESC = (
    "Hospital with 200 beds, 50 doctors, and 1000 patients. "
    "We need to optimize treatment scheduling and predict patient outcomes."
)

MILITARY_DESC = (
    "Military base with 3 supply depots and 20 transport vehicles. "
    "Troops face threat conditions. Optimize deployment and logistics."
)

SPORTS_DESC = (
    "Marathon runner training for a race. Track heart rate zones, "
    "VO2max data, and pace. Goal is to optimize training schedule."
)

ENVIRONMENT_DESC = (
    "Environmental monitoring station tracking river pollution levels, "
    "water quality sensors, and flood risk. Predict ecosystem impact."
)

SHORT_VAGUE_DESC = "thing"

DETAILED_HEALTHCARE_DESC = (
    "A 500-bed hospital with an oncology department treating 300 cancer patients. "
    "Each patient has a tumor profile including size, stage, and genomic markers. "
    "50 oncologists administer chemotherapy drugs with known dosage protocols. "
    "We need to optimize treatment plans to maximize patient survival rates "
    "while minimizing side effects and hospital resource utilization."
)


# ===========================================================================
# TestTwinGeneration - Engine-level tests (~8 tests)
# ===========================================================================


class TestTwinGeneration:
    """Tests for the TwinGenerator engine."""

    def test_generate_healthcare_twin(self, generator: TwinGenerator):
        """Healthcare description produces a twin with a valid twin_id."""
        result = generator.generate(HEALTHCARE_DESC)
        assert isinstance(result, GenerationResult)
        assert result.twin_id is not None and len(result.twin_id) > 0

    def test_generate_military_twin(self, generator: TwinGenerator):
        """Military description produces a twin with a valid twin_id."""
        result = generator.generate(MILITARY_DESC)
        assert result.twin_id is not None and len(result.twin_id) > 0

    def test_generate_sports_twin(self, generator: TwinGenerator):
        """Sports description produces a twin with a valid twin_id."""
        result = generator.generate(SPORTS_DESC)
        assert result.twin_id is not None and len(result.twin_id) > 0

    def test_generate_environment_twin(self, generator: TwinGenerator):
        """Environment description produces a twin with a valid twin_id."""
        result = generator.generate(ENVIRONMENT_DESC)
        assert result.twin_id is not None and len(result.twin_id) > 0

    def test_short_vague_input_yields_draft(self, generator: TwinGenerator):
        """Short vague input results in DRAFT status or low confidence."""
        result = generator.generate(SHORT_VAGUE_DESC)
        # The extractor should give low confidence for vague input,
        # which means either DRAFT status or low extraction confidence.
        is_draft = result.status == TwinStatus.DRAFT
        low_confidence = (
            result.quantum_metrics.get("extraction_confidence", 0) < 0.5
        )
        assert is_draft or low_confidence, (
            f"Expected DRAFT or low confidence for vague input, "
            f"got status={result.status}, metrics={result.quantum_metrics}"
        )

    def test_detailed_input_higher_confidence(self, generator: TwinGenerator):
        """Detailed input yields higher extraction confidence than vague input."""
        detailed = generator.generate(DETAILED_HEALTHCARE_DESC)
        vague = generator.generate(SHORT_VAGUE_DESC)

        detailed_conf = detailed.quantum_metrics.get("extraction_confidence", 0)
        vague_conf = vague.quantum_metrics.get("extraction_confidence", 0)

        # If the vague input was so low it became DRAFT (no metrics), that counts.
        if vague.status == TwinStatus.DRAFT and not vague.quantum_metrics:
            vague_conf = 0.0

        assert detailed_conf > vague_conf, (
            f"Detailed confidence ({detailed_conf}) should exceed "
            f"vague confidence ({vague_conf})"
        )

    def test_generation_time_is_reasonable(self, generator: TwinGenerator):
        """Twin generation completes in under 5 seconds."""
        start = time.time()
        generator.generate(HEALTHCARE_DESC)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Generation took {elapsed:.2f}s, expected < 5s"

    def test_generated_twin_has_extracted_system_with_entities(
        self, generator: TwinGenerator
    ):
        """Generated twin contains an ExtractedSystem with at least one entity."""
        result = generator.generate(HEALTHCARE_DESC)
        assert result.extracted_system is not None
        assert isinstance(result.extracted_system, ExtractedSystem)
        assert len(result.extracted_system.entities) > 0


# ===========================================================================
# TestFullAPILifecycle - REST API tests (~10 tests)
# ===========================================================================


class TestFullAPILifecycle:
    """Tests for the full REST API lifecycle."""

    def test_create_twin(self, client: TestClient):
        """POST /api/twins/ creates a twin and returns 201."""
        resp = client.post("/api/twins/", json={
            "name": "Test Hospital Twin",
            "description": "A hospital digital twin for testing",
            "domain": "healthcare",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["name"] == "Test Hospital Twin"
        assert data["status"] == "draft"

    def test_get_twin_by_id(self, client: TestClient):
        """GET /api/twins/{id} retrieves a previously created twin."""
        create = client.post("/api/twins/", json={
            "name": "Retrieve Me",
            "description": "Twin to retrieve",
        })
        twin_id = create.json()["id"]

        resp = client.get(f"/api/twins/{twin_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == twin_id
        assert resp.json()["name"] == "Retrieve Me"

    def test_list_twins(self, client: TestClient):
        """GET /api/twins/ returns a list that includes the created twin."""
        create = client.post("/api/twins/", json={
            "name": "Listed Twin",
            "description": "Should appear in listing",
        })
        twin_id = create.json()["id"]

        resp = client.get("/api/twins/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        ids = [t["id"] for t in data]
        assert twin_id in ids

    def test_conversation_creates_twin(self, client: TestClient):
        """POST /api/conversation with no twin_id creates a new twin."""
        resp = client.post("/api/conversation/", json={
            "message": HEALTHCARE_DESC,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "twin_id" in data
        assert data["twin_id"] is not None
        assert "message" in data
        assert "twin_status" in data

    def test_conversation_second_message_accumulates_entities(
        self, client: TestClient
    ):
        """Second conversation message builds on the first; entities accumulate."""
        # First message
        r1 = client.post("/api/conversation/", json={
            "message": "Hospital with 200 beds and 50 doctors",
        })
        twin_id = r1.json()["twin_id"]
        entities_1 = []
        if r1.json().get("extracted_info") and r1.json()["extracted_info"].get("entities"):
            entities_1 = r1.json()["extracted_info"]["entities"]

        # Second message adds more detail
        r2 = client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": "There are also 1000 patients who need treatment optimization",
        })
        assert r2.status_code == 200
        entities_2 = []
        if r2.json().get("extracted_info") and r2.json()["extracted_info"].get("entities"):
            entities_2 = r2.json()["extracted_info"]["entities"]

        # Second round should have at least as many entities
        assert len(entities_2) >= len(entities_1), (
            f"Entities should accumulate: {len(entities_1)} -> {len(entities_2)}"
        )

    def test_get_conversation_history(self, client: TestClient):
        """GET /api/conversation/{twin_id}/history returns message list."""
        # Create conversation
        r1 = client.post("/api/conversation/", json={
            "message": "Create a sports twin for marathon training",
        })
        twin_id = r1.json()["twin_id"]

        # Retrieve history
        resp = client.get(f"/api/conversation/{twin_id}/history")
        assert resp.status_code == 200
        messages = resp.json()
        assert isinstance(messages, list)
        # Should have at least the user message and the assistant reply
        assert len(messages) >= 2
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_update_twin_status(self, client: TestClient):
        """PATCH /api/twins/{id} can update the twin status."""
        create = client.post("/api/twins/", json={
            "name": "Status Update Twin",
            "description": "Testing status transitions",
        })
        twin_id = create.json()["id"]
        assert create.json()["status"] == "draft"

        resp = client.patch(f"/api/twins/{twin_id}", json={
            "status": "active",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_create_twin_missing_fields_returns_422(self, client: TestClient):
        """POST /api/twins/ with missing required fields returns 422."""
        # Missing name and description
        resp = client.post("/api/twins/", json={})
        assert resp.status_code == 422

    def test_twin_listing_returns_list_format(self, client: TestClient):
        """GET /api/twins/ always returns a JSON list (even if empty)."""
        resp = client.get("/api/twins/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_get_nonexistent_twin_returns_404(self, client: TestClient):
        """GET /api/twins/{bad_id} returns 404 for an unknown twin."""
        resp = client.get("/api/twins/nonexistent-id-12345")
        assert resp.status_code == 404


# ===========================================================================
# TestCrossDomainGeneration - parametrized (~4 tests)
# ===========================================================================


DOMAIN_CASES = [
    ("healthcare", HEALTHCARE_DESC),
    ("military", MILITARY_DESC),
    ("sports", SPORTS_DESC),
    ("environment", ENVIRONMENT_DESC),
]


class TestCrossDomainGeneration:
    """Parametrized tests proving the same TwinGenerator works across domains."""

    @pytest.mark.parametrize("domain,description", DOMAIN_CASES, ids=[d[0] for d in DOMAIN_CASES])
    def test_domain_generates_twin_with_entities(
        self, generator: TwinGenerator, domain: str, description: str
    ):
        """Each domain description generates a twin with domain-appropriate entities."""
        result = generator.generate(description)
        assert result.twin_id is not None
        assert result.extracted_system is not None
        assert len(result.extracted_system.entities) > 0, (
            f"Domain '{domain}' should extract at least one entity"
        )
        # The detected domain should match (or at least be set)
        assert result.extracted_system.domain is not None

    @pytest.mark.parametrize("domain,description", DOMAIN_CASES, ids=[d[0] for d in DOMAIN_CASES])
    def test_same_generator_instance_across_domains(
        self, generator: TwinGenerator, domain: str, description: str
    ):
        """The same TwinGenerator instance handles all domains without error."""
        result = generator.generate(description)
        assert isinstance(result, GenerationResult)
        # Should not crash; should produce a valid result for every domain
        assert result.status in (TwinStatus.DRAFT, TwinStatus.ACTIVE)


# ===========================================================================
# TestWebSocket - WebSocket connectivity (~3 tests)
# ===========================================================================


class TestWebSocket:
    """Tests for the WebSocket endpoint at /ws/{twin_id}."""

    def test_websocket_connect_and_receive_connected_message(
        self, client: TestClient
    ):
        """Connecting to /ws/{twin_id} returns a 'connected' message."""
        with client.websocket_connect("/ws/test-twin-ws-001") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert data["twin_id"] == "test-twin-ws-001"

    def test_websocket_ping_pong(self, client: TestClient):
        """Sending a ping action returns a pong response."""
        with client.websocket_connect("/ws/test-twin-ws-002") as ws:
            # Consume initial connected message
            ws.receive_json()
            # Send ping
            ws.send_json({"action": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_websocket_unknown_action_returns_ack(self, client: TestClient):
        """Sending an unknown action returns an 'ack' response."""
        with client.websocket_connect("/ws/test-twin-ws-003") as ws:
            ws.receive_json()  # connected
            ws.send_json({"action": "unknown_action"})
            data = ws.receive_json()
            assert data["type"] == "ack"
            assert data["received_action"] == "unknown_action"
