"""
WebSocket and Error Handling Tests for the Quantum Digital Twin Platform.

Validates:
- WebSocket connection lifecycle, ping/pong, unknown actions, malformed JSON
- HTTP error handling: 404s, 400s, 422 validation errors across all routers
- Health and API overview endpoints always return 200
"""

import json
import uuid

import pytest


# =============================================================================
# TestWebSocketJourney
# =============================================================================

class TestWebSocketJourney:
    """WebSocket connection lifecycle and message handling tests."""

    def test_ws_connect_receives_initial_message(self, client):
        """Connect to WebSocket and verify the initial 'connected' message."""
        twin_id = "ws-test-twin"
        with client.websocket_connect(f"/ws/{twin_id}") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert data["twin_id"] == twin_id
            assert "timestamp" in data

    def test_ws_ping_returns_pong(self, client):
        """Send a ping action and verify a pong response."""
        with client.websocket_connect("/ws/ping-test") as ws:
            # Consume initial connected message
            ws.receive_json()
            ws.send_json({"action": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data

    def test_ws_unknown_action_returns_ack(self, client):
        """Send an unrecognised action and verify an ack response."""
        with client.websocket_connect("/ws/unk-test") as ws:
            ws.receive_json()  # consume connected
            ws.send_json({"action": "do_something_unknown"})
            data = ws.receive_json()
            assert data["type"] == "ack"
            assert data["received_action"] == "do_something_unknown"
            assert "timestamp" in data

    def test_ws_malformed_json_returns_error(self, client):
        """Send invalid JSON text and verify graceful error response."""
        with client.websocket_connect("/ws/bad-json") as ws:
            ws.receive_json()  # consume connected
            ws.send_text("this is not json{{{")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON" in data["detail"]


# =============================================================================
# TestErrorHandling
# =============================================================================

class TestErrorHandling:
    """HTTP error handling across all API routers."""

    # ---- Twin errors --------------------------------------------------------

    def test_get_nonexistent_twin_returns_404(self, client):
        """GET /api/twins/{id} with a fake ID returns 404."""
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/twins/{fake_id}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_simulate_nonexistent_twin_returns_404(self, client):
        """POST /api/twins/{id}/simulate with a fake ID returns 404."""
        fake_id = str(uuid.uuid4())
        resp = client.post(
            f"/api/twins/{fake_id}/simulate",
            json={"time_steps": 10, "scenarios": 1},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_simulate_draft_twin_returns_400(self, client):
        """POST /api/twins/{id}/simulate on a DRAFT twin returns 400."""
        # Create a draft twin first (trailing slash required)
        create_resp = client.post(
            "/api/twins/",
            json={"name": "Draft Twin", "description": "A twin in draft status"},
        )
        assert create_resp.status_code == 201
        twin_id = create_resp.json()["id"]
        assert create_resp.json()["status"] == "draft"

        # Attempt to simulate
        resp = client.post(
            f"/api/twins/{twin_id}/simulate",
            json={"time_steps": 10, "scenarios": 1},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "ACTIVE" in detail or "LEARNING" in detail

    def test_query_nonexistent_twin_returns_404(self, client):
        """POST /api/twins/{id}/query with a fake ID returns 404."""
        fake_id = str(uuid.uuid4())
        resp = client.post(
            f"/api/twins/{fake_id}/query",
            json={"query": "What happens next?"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_create_twin_missing_name_returns_422(self, client):
        """POST /api/twins/ without required 'name' field returns 422."""
        resp = client.post(
            "/api/twins/",
            json={"description": "No name provided"},
        )
        assert resp.status_code == 422

    def test_create_twin_empty_description_returns_422(self, client):
        """POST /api/twins/ with empty description violates min_length=1."""
        resp = client.post(
            "/api/twins/",
            json={"name": "Good Name", "description": ""},
        )
        assert resp.status_code == 422

    # ---- Conversation errors ------------------------------------------------

    def test_conversation_nonexistent_twin_returns_404(self, client):
        """POST /api/conversation/ with a nonexistent twin_id returns 404."""
        fake_id = str(uuid.uuid4())
        resp = client.post(
            "/api/conversation/",
            json={"twin_id": fake_id, "message": "Hello"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    # ---- Benchmark errors ---------------------------------------------------

    def test_benchmark_results_nonexistent_module_returns_404(self, client):
        """GET /api/benchmark/results/{module_id} with unknown module returns 404."""
        resp = client.get("/api/benchmark/results/nonexistent_module")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    # ---- Always-available endpoints -----------------------------------------

    def test_health_check_returns_200(self, client):
        """GET /health always returns 200 with status=healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"

    def test_api_overview_returns_200(self, client):
        """GET /api returns 200 with version and sections."""
        resp = client.get("/api")
        assert resp.status_code == 200
        body = resp.json()
        assert "version" in body
        assert "sections" in body
