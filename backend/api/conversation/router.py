"""
Conversation API Router - Natural language interface for twin creation and querying.

Uses the SpaCyEnhancedExtractor which wraps SystemExtractor (regex) with
optional spaCy NER enrichment for a two-stage extraction pipeline.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.models.database import get_db, TwinModel, ConversationModel
from backend.models.schemas import (
    ConversationRequest,
    ConversationResponse,
    ExtractedSystem,
    Entity,
    Relationship,
    Rule,
    Constraint,
    TwinStatus,
    Message,
)
from backend.engine.extraction import SystemExtractor, ExtractionResult
from backend.auth.dependencies import get_current_user_optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SpaCyEnhancedExtractor: two-stage extraction (regex + spaCy NER)
# ---------------------------------------------------------------------------

class SpaCyEnhancedExtractor:
    """Two-stage entity extractor.

    Stage 1: rule-based SystemExtractor (deterministic, domain-adaptive).
    Stage 2: spaCy en_core_web_sm NER enrichment (additive only).

    Falls back gracefully to regex-only if spaCy is unavailable.
    """

    def __init__(self):
        self._regex_extractor = SystemExtractor()
        self._nlp = None
        self._spacy_attempted = False

    def _ensure_spacy(self):
        """Lazy-load spaCy model once."""
        if self._spacy_attempted:
            return self._nlp
        self._spacy_attempted = True
        try:
            from backend.ai.providers.local import _get_nlp
            self._nlp = _get_nlp()
            if self._nlp:
                logger.info("SpaCyEnhancedExtractor: spaCy NER available")
            else:
                logger.info("SpaCyEnhancedExtractor: spaCy unavailable, regex-only mode")
        except Exception as exc:
            logger.warning("SpaCyEnhancedExtractor: failed to load spaCy: %s", exc)
            self._nlp = None
        return self._nlp

    def extract(self, text: str, existing_system=None) -> ExtractionResult:
        """Run two-stage extraction: regex first, then spaCy NER enrichment."""
        # Stage 1: rule-based extraction (preserves all existing behavior)
        result = self._regex_extractor.extract(text, existing_system)

        # Stage 2: spaCy NER enrichment (additive only)
        nlp = self._ensure_spacy()
        if nlp is None:
            return result

        try:
            doc = nlp(text)
            existing_names = {e.name.lower() for e in result.system.entities}

            for ent in doc.ents:
                if ent.label_ in ("ORG", "GPE", "PERSON", "PRODUCT", "EVENT",
                                  "QUANTITY", "CARDINAL", "DATE", "TIME",
                                  "NORP", "FAC", "LOC", "WORK_OF_ART"):
                    name_lower = ent.text.lower().strip()
                    if name_lower and name_lower not in existing_names:
                        new_entity = Entity(
                            id=f"spacy_{ent.label_.lower()}_{len(result.system.entities)}",
                            name=ent.text.strip(),
                            type=ent.label_.lower(),
                            properties={"source": "spacy_ner"},
                        )
                        result.system.entities.append(new_entity)
                        existing_names.add(name_lower)
        except Exception as exc:
            logger.warning("SpaCyEnhancedExtractor: NER enrichment failed: %s", exc)

        return result


router = APIRouter(prefix="/conversation", tags=["conversation"])

# Two-stage extractor: regex (SystemExtractor) + spaCy NER enrichment
_extractor = SpaCyEnhancedExtractor()


# =============================================================================
# Conversation Endpoints
# =============================================================================

@router.post("/", response_model=ConversationResponse)
async def send_message(
    request: ConversationRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    """
    Send a message in the conversation.

    If twin_id is None, this starts the twin creation process.
    The AI extracts system components from the user's description.
    """
    # Get or create twin
    if request.twin_id:
        db_twin = db.query(TwinModel).filter(TwinModel.id == request.twin_id).first()
        if not db_twin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Twin with id {request.twin_id} not found"
            )
    else:
        # Create new twin from first message
        db_twin = TwinModel(
            id=str(uuid.uuid4()),
            name="New Digital Twin",
            description=request.message[:500],
            status=TwinStatus.DRAFT.value,
            state={"entities": {}, "time_step": 0, "metrics": {}},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(db_twin)
        db.commit()
        db.refresh(db_twin)

    # Get or create conversation — use get-or-create pattern to prevent race conditions
    db_conversation = db.query(ConversationModel).filter(
        ConversationModel.twin_id == db_twin.id
    ).first()

    if not db_conversation:
        db_conversation = ConversationModel(
            id=str(uuid.uuid4()),
            twin_id=db_twin.id,
            messages=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(db_conversation)
        try:
            db.flush()
        except Exception:
            # Another request created the conversation concurrently — reload
            db.rollback()
            db_conversation = db.query(ConversationModel).filter(
                ConversationModel.twin_id == db_twin.id
            ).first()
            if not db_conversation:
                logger.error("Failed to create or find conversation for twin %s", db_twin.id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to initialise conversation",
                )

    # Add user message to conversation
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    messages = db_conversation.messages or []
    messages.append(user_message)

    # ---- Use the real SystemExtractor from backend.engine.extraction ----
    existing_system = None
    if db_twin.extracted_system:
        try:
            existing_system = ExtractedSystem(**db_twin.extracted_system)
        except Exception as exc:
            logger.warning(
                "Could not parse existing extracted_system for twin %s: %s",
                db_twin.id, exc,
            )
            existing_system = None

    extraction_result: ExtractionResult = _extractor.extract(
        request.message, existing_system
    )
    extracted_info = extraction_result.system

    # Determine what information is still needed (from the extractor result)
    missing_info = extraction_result.missing_info

    # Generate AI response
    if missing_info:
        response_text, questions = _generate_clarifying_response(
            extracted_info, missing_info, extraction_result.suggestions
        )
        requires_more_info = True
    else:
        response_text = _generate_twin_ready_response(extracted_info)
        questions = []
        requires_more_info = False

        # Twin generation is instantaneous — transition directly to ACTIVE
        db_twin.status = TwinStatus.ACTIVE.value

    # Add assistant message to conversation
    assistant_message = {
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    messages.append(assistant_message)

    # Update conversation and twin
    db_conversation.messages = messages
    db_conversation.updated_at = datetime.now(timezone.utc)

    db_twin.extracted_system = extracted_info.model_dump() if extracted_info else None
    db_twin.updated_at = datetime.now(timezone.utc)

    # Auto-detect domain
    if not db_twin.domain and extracted_info:
        db_twin.domain = extracted_info.domain

    db.commit()

    return ConversationResponse(
        twin_id=db_twin.id,
        message=response_text,
        extracted_info=extracted_info,
        suggestions=_generate_suggestions(extracted_info),
        twin_status=TwinStatus(db_twin.status),
        requires_more_info=requires_more_info,
        questions=questions,
    )


@router.get("/{twin_id}/history", response_model=List[Message])
async def get_conversation_history(
    twin_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    """Get conversation history for a twin."""
    db_conversation = db.query(ConversationModel).filter(
        ConversationModel.twin_id == twin_id
    ).first()

    if not db_conversation:
        return []

    return [
        Message(
            role=msg["role"],
            content=msg["content"],
            timestamp=datetime.fromisoformat(msg["timestamp"]),
            metadata=msg.get("metadata", {}),
        )
        for msg in (db_conversation.messages or [])
    ]


# =============================================================================
# Response generation helpers
# =============================================================================

def _generate_clarifying_response(
    system: ExtractedSystem,
    missing: List[str],
    extractor_suggestions: List[str],
) -> tuple:
    """Generate a response asking for more information."""
    questions = []

    if "entities" in missing:
        questions.append("What are the main components or entities in your system?")
    if "goal" in missing:
        questions.append("What would you like to predict, optimize, or understand?")
    if "entity_properties" in missing:
        questions.append("What properties or measurements are important for each entity?")
    if "constraints" in missing:
        questions.append("Are there any constraints or limits I should know about?")

    # Append suggestions from the extractor that are not already in questions
    for suggestion in extractor_suggestions:
        if suggestion not in questions:
            questions.append(suggestion)

    extracted_summary = ""
    if system.entities:
        extracted_summary = f"\n\nSo far, I've identified: {', '.join(e.name for e in system.entities)}"
    if system.domain:
        extracted_summary += f"\nDomain: {system.domain}"

    response = f"I'm building your quantum digital twin.{extracted_summary}\n\nTo continue, I need to know:\n"
    response += "\n".join(f"\u2022 {q}" for q in questions)

    return response, questions


def _generate_twin_ready_response(system: ExtractedSystem) -> str:
    """Generate a response when twin is ready to be generated."""
    entities_str = ", ".join(e.name for e in system.entities)

    return f"""Your Quantum Digital Twin is ready to be generated!

**System Summary:**
- Domain: {system.domain}
- Entities: {entities_str}
- Relationships: {len(system.relationships)} identified
- Goal: {system.goal}

I'm now generating the quantum circuits and preparing the simulation environment.
This typically takes a few seconds.

Once ready, you'll be able to:
\u2022 Run simulations across thousands of scenarios simultaneously
\u2022 Ask "what if" questions
\u2022 Optimize for your goals
\u2022 Explore possible futures

Generating your twin now..."""


def _generate_suggestions(system: ExtractedSystem) -> List[str]:
    """Generate suggestions based on extracted system."""
    suggestions = []

    if system.goal == "optimize":
        suggestions.append("Try: 'What's the optimal strategy?'")
    if system.domain == "healthcare":
        suggestions.append("Try: 'Compare treatment options'")
    if system.domain == "sports":
        suggestions.append("Try: 'What's my optimal pacing strategy?'")

    return suggestions
