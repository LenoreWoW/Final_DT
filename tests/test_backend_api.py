"""
Tests for Backend API - Phase 1

Tests cover:
- Health check
- Twin CRUD operations
- Conversation flow
- System extraction
"""

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.models.database import Base, get_db
from backend.models.schemas import TwinStatus


# =============================================================================
# Test Fixtures
# =============================================================================

# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client():
    """Create test client with fresh database for each test.

    NOTE: This fixture intentionally shadows the conftest ``client`` fixture
    because this module needs its own dedicated in-memory engine and session
    setup that is independent of the shared conftest database.
    """
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Override dependency
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    # Cleanup - drop tables and remove only *our* override
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.pop(get_db, None)


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns platform info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Quantum Digital Twin Platform"
        assert data["version"] == "2.0.0"
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"

    def test_api_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        
        data = response.json()
        assert "sections" in data
        assert "builder" in data["sections"]
        assert "showcase" in data["sections"]


# =============================================================================
# Twin CRUD Tests
# =============================================================================

class TestTwinCRUD:
    """Tests for twin CRUD operations."""

    def test_create_twin(self, client):
        """Test creating a new twin."""
        twin_data = {
            "name": "Marathon Runner Twin",
            "description": "Digital twin for marathon race optimization",
            "domain": "sports"
        }
        
        response = client.post("/api/twins/", json=twin_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["name"] == "Marathon Runner Twin"
        assert data["description"] == "Digital twin for marathon race optimization"
        assert data["domain"] == "sports"
        assert data["status"] == TwinStatus.DRAFT.value
        assert "id" in data
        assert "created_at" in data

    def test_create_twin_minimal(self, client):
        """Test creating a twin with minimal data."""
        twin_data = {
            "name": "Test Twin",
            "description": "A test digital twin"
        }
        
        response = client.post("/api/twins/", json=twin_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["name"] == "Test Twin"
        assert data["domain"] is None  # Not provided

    def test_create_twin_validation(self, client):
        """Test twin creation validation."""
        # Missing required fields
        response = client.post("/api/twins/", json={})
        assert response.status_code == 422  # Validation error

        # Empty name
        response = client.post("/api/twins/", json={"name": "", "description": "test"})
        assert response.status_code == 422

    def test_list_twins(self, client):
        """Test listing all twins."""
        # Create some twins
        for i in range(3):
            client.post("/api/twins/", json={
                "name": f"Twin {i}",
                "description": f"Description {i}"
            })
        
        response = client.get("/api/twins/")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 3

    def test_list_twins_filter_by_status(self, client):
        """Test filtering twins by status."""
        # Create a twin
        client.post("/api/twins/", json={
            "name": "Draft Twin",
            "description": "A draft twin"
        })
        
        # Filter by DRAFT status
        response = client.get("/api/twins/?status=draft")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "draft"
        
        # Filter by ACTIVE status (should be empty)
        response = client.get("/api/twins/?status=active")
        assert response.status_code == 200
        assert len(response.json()) == 0

    def test_get_twin(self, client):
        """Test getting a specific twin."""
        # Create a twin
        create_response = client.post("/api/twins/", json={
            "name": "Get Me Twin",
            "description": "A twin to get"
        })
        twin_id = create_response.json()["id"]
        
        # Get it
        response = client.get(f"/api/twins/{twin_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == twin_id
        assert data["name"] == "Get Me Twin"

    def test_get_twin_not_found(self, client):
        """Test getting a non-existent twin."""
        response = client.get("/api/twins/nonexistent-id")
        assert response.status_code == 404

    def test_update_twin(self, client):
        """Test updating a twin."""
        # Create a twin
        create_response = client.post("/api/twins/", json={
            "name": "Original Name",
            "description": "Original description"
        })
        twin_id = create_response.json()["id"]
        
        # Update it
        response = client.patch(f"/api/twins/{twin_id}", json={
            "name": "Updated Name"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Original description"  # Unchanged

    def test_update_twin_status(self, client):
        """Test updating twin status."""
        # Create a twin
        create_response = client.post("/api/twins/", json={
            "name": "Status Twin",
            "description": "Testing status update"
        })
        twin_id = create_response.json()["id"]
        
        # Update status
        response = client.patch(f"/api/twins/{twin_id}", json={
            "status": "active"
        })
        assert response.status_code == 200
        assert response.json()["status"] == "active"

    def test_delete_twin(self, client):
        """Test deleting a twin."""
        # Create a twin
        create_response = client.post("/api/twins/", json={
            "name": "Delete Me",
            "description": "Going away"
        })
        twin_id = create_response.json()["id"]
        
        # Delete it
        response = client.delete(f"/api/twins/{twin_id}")
        assert response.status_code == 204
        
        # Verify it's gone
        response = client.get(f"/api/twins/{twin_id}")
        assert response.status_code == 404


# =============================================================================
# Conversation Tests
# =============================================================================

class TestConversation:
    """Tests for conversation endpoints."""

    def test_start_new_conversation(self, client):
        """Test starting a new conversation (creates twin)."""
        response = client.post("/api/conversation/", json={
            "message": "I'm running a marathon in 8 weeks with two major hills"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "twin_id" in data
        assert "message" in data
        assert data["twin_status"] in ["draft", "generating"]
        
        # Twin should be created
        twin_id = data["twin_id"]
        twin_response = client.get(f"/api/twins/{twin_id}")
        assert twin_response.status_code == 200

    def test_continue_conversation(self, client):
        """Test continuing an existing conversation."""
        # Start conversation
        start_response = client.post("/api/conversation/", json={
            "message": "I need help with my marathon training"
        })
        twin_id = start_response.json()["twin_id"]
        
        # Continue conversation
        response = client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": "The race has two major hills at mile 8 and mile 20"
        })
        assert response.status_code == 200
        assert response.json()["twin_id"] == twin_id

    def test_conversation_extracts_domain(self, client):
        """Test that conversation extracts domain from message."""
        # Healthcare domain
        response = client.post("/api/conversation/", json={
            "message": "I was diagnosed with cancer and need help with treatment options"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data.get("extracted_info") is not None, "extracted_info should always be present"
        assert data["extracted_info"]["domain"] == "healthcare"

    def test_get_conversation_history(self, client):
        """Test getting conversation history."""
        # Start conversation
        start_response = client.post("/api/conversation/", json={
            "message": "Hello, I need a digital twin"
        })
        twin_id = start_response.json()["twin_id"]
        
        # Get history
        response = client.get(f"/api/conversation/{twin_id}/history")
        assert response.status_code == 200
        
        history = response.json()
        assert len(history) >= 2  # User message + AI response

    def test_conversation_with_invalid_twin(self, client):
        """Test conversation with non-existent twin."""
        response = client.post("/api/conversation/", json={
            "twin_id": "nonexistent-id",
            "message": "Hello?"
        })
        assert response.status_code == 404


# =============================================================================
# Simulation Tests
# =============================================================================

class TestSimulation:
    """Tests for simulation endpoints."""

    def test_simulation_requires_active_twin(self, client):
        """Test that simulation requires an active twin."""
        # Create a draft twin
        create_response = client.post("/api/twins/", json={
            "name": "Draft Twin",
            "description": "Still in draft"
        })
        twin_id = create_response.json()["id"]
        
        # Try to simulate (should fail - not active)
        response = client.post(f"/api/twins/{twin_id}/simulate", json={
            "time_steps": 100
        })
        assert response.status_code == 400

    def test_simulation_on_active_twin(self, client):
        """Test running simulation on an active twin."""
        # Create and activate a twin
        create_response = client.post("/api/twins/", json={
            "name": "Active Twin",
            "description": "Ready for simulation"
        })
        twin_id = create_response.json()["id"]
        
        # Activate it
        client.patch(f"/api/twins/{twin_id}", json={"status": "active"})
        
        # Run simulation
        response = client.post(f"/api/twins/{twin_id}/simulate", json={
            "time_steps": 100,
            "scenarios": 10
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["twin_id"] == twin_id
        assert data["time_steps"] == 100
        assert data["scenarios_run"] == 10
        assert "quantum_advantage" in data


# =============================================================================
# Query Tests
# =============================================================================

class TestQuery:
    """Tests for query endpoints."""

    def test_query_twin(self, client):
        """Test querying a twin."""
        # Create a twin
        create_response = client.post("/api/twins/", json={
            "name": "Query Twin",
            "description": "Testing queries"
        })
        twin_id = create_response.json()["id"]
        
        # Query it
        response = client.post(f"/api/twins/{twin_id}/query", json={
            "query": "What happens in 6 months?"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["twin_id"] == twin_id
        assert data["query_type"] == "prediction"  # Auto-detected

    def test_query_type_detection(self, client):
        """Test automatic query type detection."""
        # Create a twin
        create_response = client.post("/api/twins/", json={
            "name": "Query Type Twin",
            "description": "Testing query types"
        })
        twin_id = create_response.json()["id"]
        
        test_cases = [
            ("What happens next month?", "prediction"),
            ("What's the best strategy?", "optimization"),
            ("Show me all possibilities", "exploration"),
            ("What if I had done X instead?", "counterfactual"),
            ("Why did this happen?", "understanding"),
            ("Compare option A vs B", "comparison"),
        ]
        
        for query, expected_type in test_cases:
            response = client.post(f"/api/twins/{twin_id}/query", json={
                "query": query
            })
            assert response.status_code == 200
            assert response.json()["query_type"] == expected_type, f"Failed for: {query}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_twin_lifecycle(self, client):
        """Test complete twin lifecycle: create -> conversation -> simulate."""
        # 1. Start with conversation
        conv_response = client.post("/api/conversation/", json={
            "message": "I'm a marathon runner preparing for Boston Marathon"
        })
        assert conv_response.status_code == 200
        twin_id = conv_response.json()["twin_id"]
        
        # 2. Continue conversation to provide more info
        client.post("/api/conversation/", json={
            "twin_id": twin_id,
            "message": "I want to optimize my pacing strategy for a 3:30 finish"
        })
        
        # 3. Check twin was updated
        twin_response = client.get(f"/api/twins/{twin_id}")
        assert twin_response.status_code == 200
        
        # 4. Activate twin
        client.patch(f"/api/twins/{twin_id}", json={"status": "active"})
        
        # 5. Run simulation
        sim_response = client.post(f"/api/twins/{twin_id}/simulate", json={
            "time_steps": 100,
            "scenarios": 10
        })
        assert sim_response.status_code == 200
        assert sim_response.json()["quantum_advantage"]["speedup"] > 1
        
        # 6. Query the twin
        query_response = client.post(f"/api/twins/{twin_id}/query", json={
            "query": "What's the optimal pace for the first 10 miles?"
        })
        assert query_response.status_code == 200
        
        # 7. Clean up
        delete_response = client.delete(f"/api/twins/{twin_id}")
        assert delete_response.status_code == 204


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

