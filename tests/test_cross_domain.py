"""
Cross-Domain Twin Generation Tests.

Tests that the platform correctly creates digital twins across all 4 thesis
domains (healthcare, military, sports, environment), extracts domain-specific
entities, and isolates entities between twins in different domains.

Usage:
    cd /Users/hassanalsahli/Desktop/Final_DT
    venv/bin/python -m pytest tests/test_cross_domain.py -v
"""

import pytest


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DOMAIN_DESCRIPTIONS = {
    "healthcare": (
        "Hospital with 200 beds, 50 doctors, 1000 patients "
        "needing scheduling optimization"
    ),
    "military": (
        "Military base with a battalion of troops and supply depots "
        "under threat conditions from enemy adversary forces"
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

DOMAINS = list(DOMAIN_DESCRIPTIONS.keys())


# ===================================================================
# Test class: Twin creation per domain
# ===================================================================

class TestDomainTwinCreation:
    """Create a twin in each domain and verify extraction."""

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_create_twin_returns_200(self, client, domain):
        """POST /api/conversation/ with a domain description creates a twin."""
        resp = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS[domain],
        })
        assert resp.status_code == 200, (
            f"Expected 200 for {domain}, got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "twin_id" in data
        assert data["twin_id"], f"twin_id should be non-empty for {domain}"

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_extracted_domain_is_correct(self, client, domain):
        """The extracted_info.domain matches the expected domain."""
        resp = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS[domain],
        })
        assert resp.status_code == 200
        data = resp.json()

        extracted = data.get("extracted_info")
        assert extracted is not None, (
            f"extracted_info should be present for {domain}"
        )
        assert extracted.get("domain") == domain, (
            f"Expected domain='{domain}', got '{extracted.get('domain')}'"
        )

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_entities_extracted(self, client, domain):
        """At least one entity is extracted from the domain description."""
        resp = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS[domain],
        })
        assert resp.status_code == 200
        data = resp.json()

        extracted = data.get("extracted_info", {})
        entities = extracted.get("entities", [])
        assert len(entities) > 0, (
            f"Expected at least 1 entity for {domain}, got {len(entities)}"
        )


# ===================================================================
# Test class: Cross-domain entity isolation
# ===================================================================

class TestCrossDomainIsolation:
    """Verify that twins in different domains have isolated entities."""

    def test_healthcare_and_military_entities_differ(self, client):
        """Entities from a healthcare twin should not appear in a military twin."""
        resp_health = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS["healthcare"],
        })
        resp_military = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS["military"],
        })

        assert resp_health.status_code == 200
        assert resp_military.status_code == 200

        health_entities = {
            e["name"] if isinstance(e, dict) else str(e)
            for e in resp_health.json().get("extracted_info", {}).get("entities", [])
        }
        military_entities = {
            e["name"] if isinstance(e, dict) else str(e)
            for e in resp_military.json().get("extracted_info", {}).get("entities", [])
        }

        # The two entity sets should not be identical
        assert health_entities != military_entities, (
            "Healthcare and military twins should have different entities"
        )

    def test_sports_and_environment_entities_differ(self, client):
        """Entities from a sports twin should not appear in an environment twin."""
        resp_sports = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS["sports"],
        })
        resp_env = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS["environment"],
        })

        assert resp_sports.status_code == 200
        assert resp_env.status_code == 200

        sports_entities = {
            e["name"] if isinstance(e, dict) else str(e)
            for e in resp_sports.json().get("extracted_info", {}).get("entities", [])
        }
        env_entities = {
            e["name"] if isinstance(e, dict) else str(e)
            for e in resp_env.json().get("extracted_info", {}).get("entities", [])
        }

        assert sports_entities != env_entities, (
            "Sports and environment twins should have different entities"
        )

    def test_twins_have_different_ids(self, client):
        """Each domain twin should receive a unique twin_id."""
        twin_ids = set()
        for domain in DOMAINS:
            resp = client.post("/api/conversation/", json={
                "message": DOMAIN_DESCRIPTIONS[domain],
            })
            assert resp.status_code == 200
            twin_ids.add(resp.json()["twin_id"])

        assert len(twin_ids) == len(DOMAINS), (
            f"Expected {len(DOMAINS)} unique twin_ids, got {len(twin_ids)}"
        )


# ===================================================================
# Test class: Domain detection accuracy
# ===================================================================

class TestDomainDetectionAccuracy:
    """Verify each description is classified into the correct domain."""

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_domain_detection(self, client, domain):
        """The platform correctly detects the domain from the description."""
        resp = client.post("/api/conversation/", json={
            "message": DOMAIN_DESCRIPTIONS[domain],
        })
        assert resp.status_code == 200
        data = resp.json()

        detected_domain = data.get("extracted_info", {}).get("domain")
        assert detected_domain == domain, (
            f"Expected domain '{domain}' but detected '{detected_domain}'"
        )
