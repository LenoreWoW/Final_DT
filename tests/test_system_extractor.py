"""
Tests for SystemExtractor — NLP extraction pipeline.

Covers entity extraction, domain detection, goal extraction,
confidence scoring, and accumulative extraction across all thesis domains.
"""

import pytest

from backend.engine.extraction.system_extractor import (
    SystemExtractor,
    ExtractionResult,
    DomainType,
    GoalType,
    extract_system,
)
from backend.models.schemas import ExtractedSystem, Entity


# ---------------------------------------------------------------------------
# TestEntityExtraction
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    """Validate entity extraction across domains, edge cases, and structure."""

    def test_healthcare_entities(self, extractor: SystemExtractor):
        """Healthcare keywords should produce patient/treatment entities."""
        result = extractor.extract(
            "A patient with stage-3 cancer receiving chemotherapy treatment at a hospital"
        )
        entities = result.system.entities
        assert len(entities) >= 2
        types = [e.type for e in entities]
        assert "patient" in types
        assert "tumor" in types or "treatment" in types

    def test_military_entities(self, extractor: SystemExtractor):
        """Military descriptions should produce unit/personnel entities."""
        result = extractor.extract(
            "A battalion of 500 troops deployed to defend a terrain region against enemy forces"
        )
        entities = result.system.entities
        assert len(entities) >= 2
        types = [e.type for e in entities]
        assert "unit" in types or "personnel" in types

    def test_sports_entities(self, extractor: SystemExtractor):
        """Sports descriptions should produce athlete/race entities."""
        result = extractor.extract(
            "An athlete running a marathon race with heart-rate monitoring"
        )
        entities = result.system.entities
        assert len(entities) >= 1
        types = [e.type for e in entities]
        assert "athlete" in types or "race" in types

    def test_environment_entities(self, extractor: SystemExtractor):
        """Environment descriptions should produce ecosystem/species entities."""
        result = extractor.extract(
            "A wildfire spreading through a forest ecosystem threatening local species"
        )
        entities = result.system.entities
        assert len(entities) >= 2
        types = [e.type for e in entities]
        assert "fire" in types or "ecosystem" in types

    def test_entity_has_type_and_name(self, extractor: SystemExtractor):
        """Every extracted entity must have non-empty type and name."""
        result = extractor.extract(
            "Hospital with 200 beds treating cancer patients with new drug therapies"
        )
        for entity in result.system.entities:
            assert entity.type, "Entity type must not be empty"
            assert entity.name, "Entity name must not be empty"
            assert isinstance(entity.type, str)
            assert isinstance(entity.name, str)

    def test_relationship_extraction(self, extractor: SystemExtractor):
        """When 2+ entities exist, implicit relationships should be generated."""
        result = extractor.extract(
            "A patient receiving treatment at a hospital"
        )
        assert len(result.system.entities) >= 2
        assert len(result.system.relationships) >= 1
        rel = result.system.relationships[0]
        assert rel.source_id
        assert rel.target_id
        assert rel.type

    def test_constraint_extraction(self, extractor: SystemExtractor):
        """Budget and time constraints should be extracted from text."""
        result = extractor.extract(
            "Hospital with budget of $500000 must finish within 30 days"
        )
        assert len(result.system.constraints) >= 1
        constraint_types = [c.type for c in result.system.constraints]
        assert "budget" in constraint_types or "time" in constraint_types

    def test_empty_input(self, extractor: SystemExtractor):
        """Empty string should return an ExtractionResult with no entities."""
        result = extractor.extract("")
        assert isinstance(result, ExtractionResult)
        assert len(result.system.entities) == 0

    def test_very_long_input(self, extractor: SystemExtractor):
        """Extractor should handle a very long input without error."""
        long_text = "patient treatment hospital drug " * 500
        result = extractor.extract(long_text)
        assert isinstance(result, ExtractionResult)
        assert len(result.system.entities) >= 1


# ---------------------------------------------------------------------------
# TestDomainDetection
# ---------------------------------------------------------------------------

class TestDomainDetection:
    """Validate domain classification for all four thesis domains + edge cases."""

    @pytest.mark.parametrize(
        "description, expected_domain",
        [
            # Healthcare (3 descriptions)
            (
                "A cancer patient receiving chemotherapy at the hospital",
                DomainType.HEALTHCARE,
            ),
            (
                "Tumor diagnosis with drug interaction analysis for medical treatment",
                DomainType.HEALTHCARE,
            ),
            (
                "Genomic analysis for oncology clinical trials with vaccine research",
                DomainType.HEALTHCARE,
            ),
            # Military (3 descriptions)
            (
                "A battalion of soldiers defending against enemy combat forces",
                DomainType.MILITARY,
            ),
            (
                "Tactical reconnaissance and intelligence deployment of troops",
                DomainType.MILITARY,
            ),
            (
                "Military convoy transporting ammunition to a strategic depot",
                DomainType.MILITARY,
            ),
            # Sports (2 descriptions)
            (
                "An athlete training for a marathon race to improve stamina and pace",
                DomainType.SPORTS,
            ),
            (
                "Basketball team competing in the championship with fitness tracking",
                DomainType.SPORTS,
            ),
            # Environment (3 descriptions)
            (
                "A wildfire spreading across forest ecosystem threatening wildlife species",
                DomainType.ENVIRONMENT,
            ),
            (
                "River pollution monitoring with flood risk and deforestation tracking",
                DomainType.ENVIRONMENT,
            ),
            (
                "Carbon emissions climate monitoring for biodiversity conservation",
                DomainType.ENVIRONMENT,
            ),
        ],
        ids=[
            "healthcare-chemo",
            "healthcare-tumor",
            "healthcare-genomic",
            "military-battalion",
            "military-recon",
            "military-convoy",
            "sports-marathon",
            "sports-basketball",
            "environment-wildfire",
            "environment-river",
            "environment-carbon",
        ],
    )
    def test_domain_detection(
        self,
        extractor: SystemExtractor,
        description: str,
        expected_domain: DomainType,
    ):
        """Top-confidence domain should match the expected domain."""
        result = extractor.extract(description)
        # The domain stored in the system is the detected domain value
        assert result.system.domain == expected_domain.value
        # domain_confidence dict should contain the expected domain
        assert expected_domain.value in result.domain_confidence
        # The expected domain should have the highest confidence
        top_domain = max(result.domain_confidence, key=result.domain_confidence.get)
        assert top_domain == expected_domain.value


# ---------------------------------------------------------------------------
# TestGoalExtraction
# ---------------------------------------------------------------------------

class TestGoalExtraction:
    """Validate goal detection from user intent descriptions."""

    def test_optimize_goal(self, extractor: SystemExtractor):
        """'optimize' / 'maximize' keywords should map to optimize goal."""
        result = extractor.extract(
            "I want to optimize the treatment schedule for patients"
        )
        assert result.system.goal == GoalType.OPTIMIZE.value

    def test_predict_goal(self, extractor: SystemExtractor):
        """'predict' / 'forecast' keywords should map to predict goal."""
        result = extractor.extract(
            "Predict what will happen to the wildfire spread over the next 48 hours"
        )
        assert result.system.goal == GoalType.PREDICT.value

    def test_understand_goal(self, extractor: SystemExtractor):
        """'understand' / 'why' keywords should map to understand goal."""
        result = extractor.extract(
            "I want to understand why the athlete performance drops after mile 20"
        )
        assert result.system.goal == GoalType.UNDERSTAND.value

    def test_no_goal_detected(self, extractor: SystemExtractor):
        """When no goal keywords are present, goal should be None."""
        result = extractor.extract(
            "A hospital with 200 beds and 50 doctors"
        )
        assert result.system.goal is None


# ---------------------------------------------------------------------------
# TestConfidenceAndAccumulation
# ---------------------------------------------------------------------------

class TestConfidenceAndAccumulation:
    """Validate confidence scoring, accumulation, and missing info detection."""

    def test_confidence_increases_with_detail(self, extractor: SystemExtractor):
        """More system components should produce higher confidence."""
        sparse = extractor.extract("A patient at a hospital")
        detailed = extractor.extract(
            "A cancer patient receiving drug treatment at a hospital. "
            "I want to optimize the treatment schedule. "
            "Budget of $100000, must finish within 60 days."
        )
        assert detailed.confidence > sparse.confidence

    def test_accumulative_extraction(self, extractor: SystemExtractor):
        """Passing existing_system should accumulate entities across calls."""
        first = extractor.extract(
            "A patient receiving treatment at a hospital"
        )
        first_count = len(first.system.entities)
        assert first_count >= 2

        second = extractor.extract(
            "The drug has interactions and the tumor is stage 3",
            existing_system=first.system,
        )
        assert len(second.system.entities) >= first_count

    def test_accumulative_preserves_domain(self, extractor: SystemExtractor):
        """The domain from the first extraction should persist in accumulation."""
        first = extractor.extract(
            "A cancer patient at a hospital"
        )
        assert first.system.domain == DomainType.HEALTHCARE.value

        second = extractor.extract(
            "The athlete is also a patient",
            existing_system=first.system,
        )
        # Domain should be preserved from the first extraction
        assert second.system.domain == DomainType.HEALTHCARE.value

    def test_missing_info_detection(self, extractor: SystemExtractor):
        """When entities or goal are missing, missing_info should list them."""
        result = extractor.extract("")
        assert "entities" in result.missing_info
        assert "goal" in result.missing_info

    def test_missing_info_cleared_when_complete(self, extractor: SystemExtractor):
        """A fully-specified system should have no missing_info entries."""
        result = extractor.extract(
            "A cancer patient receiving drug treatment at a hospital. "
            "I want to optimize the therapy schedule."
        )
        assert "entities" not in result.missing_info
        assert "goal" not in result.missing_info
