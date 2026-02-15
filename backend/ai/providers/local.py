"""
Local AI Provider -- spaCy + Rule-Based NLP.

This is the DEFAULT provider that requires no API keys or external services.
It combines:
  - spaCy (en_core_web_sm) for NER and linguistic analysis
  - Rule-based keyword matching for problem-type classification
  - Domain detection via keyword scoring
  - Conversation state machine for guided twin creation
  - Integration with backend.engine.extraction.SystemExtractor for
    structured entity/relationship/constraint extraction

Designed to handle these conversations naturally:
  - "I want to find optimal treatments for cancer patients"
    -> healthcare, optimization + classification
  - "Simulate supply route logistics for a military operation"
    -> military, optimization + simulation
  - "Optimize my marathon race strategy"
    -> sports, optimization
  - "Model flood impact on city infrastructure"
    -> environment, simulation + anomaly_detection
"""

import re
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import AIProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Problem-type classification
# ---------------------------------------------------------------------------

class ProblemType(str, Enum):
    OPTIMIZATION = "OPTIMIZATION"
    CLASSIFICATION = "CLASSIFICATION"
    SIMULATION = "SIMULATION"
    FORECASTING = "FORECASTING"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"


# Mapping: problem type -> recommended quantum algorithm
PROBLEM_TO_ALGORITHM = {
    ProblemType.OPTIMIZATION: "QAOA",
    ProblemType.CLASSIFICATION: "VQC/QSVM",
    ProblemType.SIMULATION: "Quantum Simulation",
    ProblemType.FORECASTING: "Tensor Network",
    ProblemType.ANOMALY_DETECTION: "Quantum Autoencoder",
}

# Keywords that signal each problem type (order matters: first match wins
# for primary, but we collect ALL matches for multi-label)
PROBLEM_KEYWORDS: Dict[ProblemType, List[str]] = {
    ProblemType.OPTIMIZATION: [
        "optimize", "optimise", "optimal", "maximize", "maximise",
        "minimize", "minimise", "best", "efficient", "efficiency",
        "route", "schedule", "allocate", "allocation", "assign",
        "logistics", "supply route", "resource",
    ],
    ProblemType.CLASSIFICATION: [
        "classify", "categorize", "categorise", "detect type",
        "identify type", "diagnose", "diagnosis", "predict type",
        "class", "label", "treatment plan", "treatment", "find optimal",
        "select", "choose", "recommend",
    ],
    ProblemType.SIMULATION: [
        "simulate", "simulation", "model", "forecast behavior",
        "dynamics", "spread", "propagation", "evolve", "evolution",
        "behavior", "behaviour", "impact", "effect",
    ],
    ProblemType.FORECASTING: [
        "forecast", "trend", "time series", "future", "predict",
        "prediction", "projection", "prognosis",
    ],
    ProblemType.ANOMALY_DETECTION: [
        "anomaly", "anomalies", "outlier", "unusual", "abnormal",
        "detect risk", "vulnerability", "threat", "risk assessment",
        "early warning", "deviation", "flood", "damage", "failure",
        "disruption", "hazard", "disaster",
    ],
}


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "healthcare": [
        "patient", "treatment", "drug", "disease", "hospital", "medical",
        "tumor", "tumour", "genomic", "epidemic", "cancer", "clinical",
        "therapy", "diagnosis", "doctor", "nurse", "symptom", "medicine",
        "health", "oncology", "virus", "infection", "vaccine", "surgery",
    ],
    "military": [
        "military", "supply route", "logistics", "base", "terrain",
        "threat", "defense", "defence", "combat", "battalion", "troops",
        "soldier", "army", "navy", "mission", "enemy", "strategic",
        "tactical", "weapon", "ammunition", "headquarters",
        "reconnaissance", "intelligence", "deployment", "operation",
    ],
    "sports": [
        "athlete", "training", "race", "pace", "marathon", "performance",
        "fitness", "runner", "player", "team", "game", "match", "score",
        "workout", "exercise", "competition", "championship", "stamina",
    ],
    "environment": [
        "flood", "earthquake", "climate", "weather", "infrastructure",
        "evacuation", "pollution", "wildfire", "ecosystem", "species",
        "population", "wildlife", "forest", "ocean", "temperature",
        "carbon", "conservation", "habitat", "biodiversity", "sustainable",
        "renewable", "acres",
    ],
    "finance": [
        "stock", "portfolio", "trading", "risk", "investment", "market",
        "asset", "return", "dividend", "bond", "equity", "hedge",
        "derivative", "currency", "cryptocurrency", "bitcoin", "bank",
    ],
}


# ---------------------------------------------------------------------------
# Conversation state machine
# ---------------------------------------------------------------------------

class ConversationState(str, Enum):
    GREETING = "greeting"
    PROBLEM_DESCRIPTION = "problem_description"
    CLARIFYING_QUESTIONS = "clarifying_questions"
    DATA_REQUEST = "data_request"
    CONFIRMATION = "confirmation"
    GENERATION = "generation"


# Patterns that indicate greetings
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))\b",
    r"^(what can you do|help me|how does this work|what is this)",
]

# Patterns that indicate confirmation
CONFIRMATION_PATTERNS = [
    r"^(yes|yeah|yep|sure|correct|right|exactly|go ahead|proceed|looks? good|that'?s? (right|correct))\b",
    r"^(generate|create|build|make|start)\b",
]


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

GREETING_RESPONSES = [
    (
        "Hello! I'm the Quantum Digital Twin assistant. I can help you build "
        "a digital twin for virtually any system -- healthcare, military "
        "logistics, sports performance, environmental modeling, finance, "
        "and more.\n\n"
        "Tell me about the system you'd like to model. For example:\n"
        "- \"I want to optimize treatment plans for cancer patients\"\n"
        "- \"Simulate supply route logistics for a military operation\"\n"
        "- \"Model flood impact on city infrastructure\"\n\n"
        "What would you like to work on?"
    ),
]

DOMAIN_FOLLOW_UPS: Dict[str, List[str]] = {
    "healthcare": [
        "What specific health conditions or treatments are you interested in modeling?",
        "How many patients or treatment options should the twin consider?",
        "Are there specific patient data points you want to incorporate (age, genetics, vitals)?",
        "Do you have existing clinical data, or should I generate synthetic scenarios?",
    ],
    "military": [
        "What is the operational objective -- logistics, combat simulation, or strategic planning?",
        "How many units or resources are involved?",
        "What terrain and environmental conditions should we model?",
        "Are there specific constraints (time limits, budget, rules of engagement)?",
    ],
    "sports": [
        "Which sport and what specific aspect of performance are we optimizing?",
        "Do you have athlete data (heart rate, pace history, training logs)?",
        "What are the key metrics you want to optimize (finish time, energy usage, injury risk)?",
        "Are there environmental factors to consider (altitude, temperature, humidity)?",
    ],
    "environment": [
        "What geographic region are we modeling?",
        "What environmental variables are most critical (water levels, wind, temperature)?",
        "What infrastructure or populations could be affected?",
        "What time horizon should the simulation cover?",
    ],
    "finance": [
        "What financial instruments are you interested in (stocks, bonds, derivatives)?",
        "What is the investment horizon and risk tolerance?",
        "How many assets should the portfolio consider?",
        "Are there regulatory constraints to model?",
    ],
}

DATA_REQUEST_RESPONSES: Dict[str, str] = {
    "healthcare": (
        "To build an accurate twin, I'll need some data. You can provide:\n"
        "- Patient demographics and medical history\n"
        "- Treatment protocols and outcomes\n"
        "- Lab results or imaging data\n\n"
        "Or I can generate realistic synthetic data to get started. "
        "Which would you prefer?"
    ),
    "military": (
        "For the military simulation twin, I can work with:\n"
        "- Unit compositions and capabilities\n"
        "- Terrain maps or descriptions\n"
        "- Supply chain data and constraints\n\n"
        "Or I can create a synthetic scenario based on your description. "
        "Would you like to provide data or use generated scenarios?"
    ),
    "sports": (
        "For performance optimization, I can use:\n"
        "- Training logs and race data\n"
        "- Physiological measurements (VO2max, heart rate zones)\n"
        "- Course profiles (elevation, surface type)\n\n"
        "Or I can model a typical scenario. What data do you have available?"
    ),
    "environment": (
        "For the environmental model, useful inputs include:\n"
        "- Geographic and topographic data\n"
        "- Historical weather/event data\n"
        "- Infrastructure maps and population data\n\n"
        "I can also generate realistic scenarios from public datasets. "
        "What information can you provide?"
    ),
    "finance": (
        "For the financial twin, I can incorporate:\n"
        "- Historical price data and market indicators\n"
        "- Portfolio composition and constraints\n"
        "- Risk parameters and investment goals\n\n"
        "Or I can use public market data for simulation. "
        "What would you like to include?"
    ),
}


# ---------------------------------------------------------------------------
# spaCy loader (lazy singleton)
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    """Lazy-load spaCy model. Falls back gracefully if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm loaded successfully")
    except Exception as e:
        logger.warning("spaCy unavailable (%s), falling back to rule-only NLP", e)
        _nlp = None
    return _nlp


# ---------------------------------------------------------------------------
# SystemExtractor integration (lazy import)
# ---------------------------------------------------------------------------

_system_extractor = None


def _get_system_extractor():
    """Lazy-load the existing SystemExtractor from the engine layer."""
    global _system_extractor
    if _system_extractor is not None:
        return _system_extractor
    try:
        from backend.engine.extraction.system_extractor import SystemExtractor
        _system_extractor = SystemExtractor()
        logger.info("SystemExtractor loaded from backend.engine.extraction")
    except ImportError:
        logger.warning(
            "SystemExtractor not available -- using local extraction only"
        )
        _system_extractor = None
    return _system_extractor


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _detect_domains(text: str) -> Dict[str, float]:
    """Score each domain by keyword match density. Returns {domain: score}."""
    text_lower = text.lower()
    scores: Dict[str, float] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[domain] = hits / len(keywords) if keywords else 0.0
    return scores


def _primary_domain(scores: Dict[str, float]) -> Optional[str]:
    """Return the top-scoring domain if it exceeds the threshold."""
    if not scores:
        return None
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0.0 else None


def _detect_problem_types(text: str) -> List[Tuple[ProblemType, float]]:
    """
    Detect all applicable problem types with confidence scores.
    Returns list of (ProblemType, confidence) sorted by confidence desc.
    """
    text_lower = text.lower()
    results: List[Tuple[ProblemType, float]] = []
    for ptype, keywords in PROBLEM_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            confidence = min(hits / 3.0, 1.0)  # 3+ hits = full confidence
            results.append((ptype, confidence))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _extract_spacy_entities(text: str) -> List[Dict[str, Any]]:
    """
    Use spaCy NER to extract named entities from text.
    Returns list of dicts: {text, label, start, end}.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    entities = []
    seen: Set[str] = set()
    for ent in doc.ents:
        key = f"{ent.text.lower()}:{ent.label_}"
        if key not in seen:
            seen.add(key)
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
    return entities


def _extract_noun_chunks(text: str) -> List[str]:
    """Extract noun chunks via spaCy for richer concept identification."""
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    chunks = []
    seen: Set[str] = set()
    for chunk in doc.noun_chunks:
        normalized = chunk.text.lower().strip()
        if normalized not in seen and len(normalized) > 2:
            seen.add(normalized)
            chunks.append(chunk.text)
    return chunks


def _infer_conversation_state(
    message: str,
    history: list,
    current_domain: Optional[str],
    current_problems: List[ProblemType],
) -> ConversationState:
    """
    Determine the current conversation state based on message content,
    history length, and what information has been gathered so far.
    """
    msg_lower = message.lower().strip()
    history_len = len(history)

    # No history and looks like a greeting
    if history_len == 0:
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, msg_lower):
                return ConversationState.GREETING

    # User confirming / requesting generation
    for pattern in CONFIRMATION_PATTERNS:
        if re.search(pattern, msg_lower):
            if current_domain and current_problems:
                return ConversationState.CONFIRMATION
            # Not enough info yet to confirm
            return ConversationState.CLARIFYING_QUESTIONS

    # Explicit generation request
    if re.search(r"\b(generate|create|build|start)\s+(the\s+)?(twin|digital twin|model|simulation)\b", msg_lower):
        if current_domain and current_problems:
            return ConversationState.GENERATION

    # If we already have domain and problems but no data discussion yet
    if current_domain and current_problems:
        # Check if data-related keywords are present
        data_keywords = ["data", "dataset", "csv", "upload", "file", "api", "database", "synthetic"]
        if any(kw in msg_lower for kw in data_keywords):
            return ConversationState.DATA_REQUEST
        # If we've been chatting a while, move to data request
        if history_len >= 4:
            return ConversationState.DATA_REQUEST
        return ConversationState.CLARIFYING_QUESTIONS

    # First substantive message with domain keywords
    if current_domain or current_problems:
        return ConversationState.PROBLEM_DESCRIPTION

    # Default: treat as problem description
    return ConversationState.PROBLEM_DESCRIPTION


def _build_response(
    state: ConversationState,
    domain: Optional[str],
    problem_types: List[Tuple[ProblemType, float]],
    spacy_entities: List[Dict[str, Any]],
    noun_chunks: List[str],
    message: str,
    history: list,
) -> Tuple[str, List[str]]:
    """
    Build a natural response and follow-up questions based on conversation state.
    Returns (response_text, follow_up_questions).
    """
    if state == ConversationState.GREETING:
        return GREETING_RESPONSES[0], []

    if state == ConversationState.GENERATION:
        algo_list = ", ".join(
            f"{PROBLEM_TO_ALGORITHM.get(pt, pt.value)} ({pt.value})"
            for pt, _ in problem_types
        )
        return (
            f"I have enough information to generate your digital twin.\n\n"
            f"**Domain:** {domain or 'general'}\n"
            f"**Quantum algorithms:** {algo_list or 'to be determined'}\n\n"
            f"I'll now build the quantum circuits and initialize the twin. "
            f"This typically takes a few seconds for the circuit design phase."
        ), []

    if state == ConversationState.CONFIRMATION:
        algo_list = ", ".join(
            f"{PROBLEM_TO_ALGORITHM.get(pt, pt.value)}"
            for pt, _ in problem_types
        )
        return (
            f"Here's what I've understood so far:\n\n"
            f"**Domain:** {domain or 'general'}\n"
            f"**Problem types:** {', '.join(pt.value for pt, _ in problem_types)}\n"
            f"**Recommended algorithms:** {algo_list}\n\n"
            f"Shall I proceed with generating the digital twin, or would you "
            f"like to refine anything?"
        ), ["Generate the twin", "I want to add more details"]

    # Build acknowledgment for problem_description and clarifying_questions
    parts: List[str] = []

    # Acknowledge what we understood
    if domain:
        domain_display = domain.replace("_", " ").title()
        parts.append(f"I can see this is a **{domain_display}** domain problem.")

    if problem_types:
        type_descriptions = []
        for pt, conf in problem_types:
            algo = PROBLEM_TO_ALGORITHM.get(pt, "")
            type_descriptions.append(f"**{pt.value}** (recommended: {algo})")
        parts.append(
            "I've identified the following problem types: "
            + ", ".join(type_descriptions)
            + "."
        )

    # Mention extracted entities if interesting
    meaningful_entities = [
        e for e in spacy_entities
        if e["label"] in ("ORG", "GPE", "PERSON", "PRODUCT", "EVENT",
                          "QUANTITY", "CARDINAL", "DATE", "TIME")
    ]
    if meaningful_entities:
        entity_strs = [f"'{e['text']}' ({e['label']})" for e in meaningful_entities[:5]]
        parts.append("I've also noted: " + ", ".join(entity_strs) + ".")

    if not parts:
        parts.append(
            "I'd like to understand your system better so I can build "
            "an accurate digital twin."
        )

    response = " ".join(parts)

    # Follow-up questions
    follow_ups: List[str] = []
    if state == ConversationState.PROBLEM_DESCRIPTION:
        if domain and domain in DOMAIN_FOLLOW_UPS:
            # Pick the first 2 follow-ups that haven't been asked yet
            asked_already = {m.get("content", "").lower() for m in history if m.get("role") == "assistant"}
            for q in DOMAIN_FOLLOW_UPS[domain]:
                if q.lower() not in asked_already:
                    follow_ups.append(q)
                if len(follow_ups) >= 2:
                    break
        if not follow_ups:
            follow_ups = [
                "Can you describe the main components of your system?",
                "What outcome are you hoping to achieve?",
            ]
        response += "\n\n" + "\n".join(f"- {q}" for q in follow_ups)

    elif state == ConversationState.CLARIFYING_QUESTIONS:
        if domain and domain in DOMAIN_FOLLOW_UPS:
            # Pick questions we haven't asked yet
            asked_already = {m.get("content", "").lower() for m in history if m.get("role") == "assistant"}
            for q in DOMAIN_FOLLOW_UPS[domain]:
                if q.lower() not in asked_already:
                    follow_ups.append(q)
                if len(follow_ups) >= 2:
                    break
        if not follow_ups:
            follow_ups = [
                "Do you have data available, or should I use synthetic data?",
                "Are there specific constraints or limits I should know about?",
            ]
        response += "\n\n" + "\n".join(f"- {q}" for q in follow_ups)

    elif state == ConversationState.DATA_REQUEST:
        if domain and domain in DATA_REQUEST_RESPONSES:
            response += "\n\n" + DATA_REQUEST_RESPONSES[domain]
        else:
            response += (
                "\n\nTo build your twin, I can either use data you provide "
                "(CSV, JSON, or API) or generate synthetic data based on your "
                "description. Which would you prefer?"
            )
        follow_ups = [
            "I'll provide data",
            "Use synthetic data for now",
        ]

    return response, follow_ups


# ---------------------------------------------------------------------------
# LocalAIProvider
# ---------------------------------------------------------------------------

class LocalAIProvider(AIProvider):
    """
    Default AI provider using spaCy NER + rule-based classification.

    No API keys required. All processing happens locally.

    Integrates with backend.engine.extraction.SystemExtractor for deep
    entity/relationship/constraint extraction from natural language.
    """

    def __init__(self):
        # Eagerly load spaCy so startup failures are loud
        nlp = _get_nlp()
        if nlp:
            logger.info("LocalAIProvider initialized with spaCy %s", nlp.meta.get("name", ""))
        else:
            logger.info("LocalAIProvider initialized in rule-only mode (spaCy unavailable)")

    # ------------------------------------------------------------------
    # AIProvider interface
    # ------------------------------------------------------------------

    async def chat(self, message: str, conversation_history: list) -> dict:
        """
        Process a single user message with spaCy NER + rule-based logic.

        All accumulated state is derived from ``conversation_history`` so
        that separate conversations never contaminate each other.
        """
        # 1. Entity extraction with spaCy
        spacy_entities = _extract_spacy_entities(message)
        noun_chunks = _extract_noun_chunks(message)

        # 2. Domain detection -- derive from the full conversation
        #    (no instance-level state; everything comes from the history)
        domain_scores = _detect_domains(message)
        full_text = " ".join(
            m.get("content", "") for m in conversation_history
            if m.get("role") == "user"
        ) + " " + message
        cumulative_scores = _detect_domains(full_text)
        msg_domain = _primary_domain(domain_scores)
        cumulative_domain = _primary_domain(cumulative_scores)
        domain = msg_domain or cumulative_domain

        # 3. Problem-type classification (from full conversation text)
        problem_types = _detect_problem_types(full_text)

        # 4. Conversation state
        state = _infer_conversation_state(
            message, conversation_history, domain, [pt for pt, _ in problem_types]
        )

        # 5. Build response
        response_text, follow_ups = _build_response(
            state=state,
            domain=domain,
            problem_types=problem_types,
            spacy_entities=spacy_entities,
            noun_chunks=noun_chunks,
            message=message,
            history=conversation_history,
        )

        primary_problem = problem_types[0][0].value if problem_types else None

        return {
            "response": response_text,
            "extracted_entities": spacy_entities,
            "problem_type": primary_problem,
            "domain": domain,
            "all_problem_types": [
                {"type": pt.value, "confidence": round(conf, 3),
                 "algorithm": PROBLEM_TO_ALGORITHM.get(pt, "")}
                for pt, conf in problem_types
            ],
            "noun_chunks": noun_chunks,
            "conversation_state": state.value,
            "follow_up_questions": follow_ups,
            "domain_scores": {k: round(v, 4) for k, v in cumulative_scores.items()},
        }

    async def extract_system(self, conversation: list) -> dict:
        """
        Analyze the full conversation to extract a structured system definition.

        Delegates to the existing SystemExtractor when available, enriching
        its output with spaCy NER and problem-type classification.
        """
        # Combine all user messages into a single description
        user_messages = [
            m.get("content", "") for m in conversation
            if m.get("role") == "user"
        ]
        full_text = " ".join(user_messages)

        # ----- SystemExtractor integration -----
        extractor = _get_system_extractor()
        extraction_result = None
        if extractor:
            try:
                extraction_result = extractor.extract(full_text)
            except Exception as e:
                logger.warning("SystemExtractor failed: %s", e)

        # ----- spaCy entity extraction -----
        spacy_entities = _extract_spacy_entities(full_text)

        # ----- Problem-type classification -----
        problem_types = _detect_problem_types(full_text)
        primary_problem = problem_types[0][0].value if problem_types else "SIMULATION"

        # ----- Domain detection -----
        domain_scores = _detect_domains(full_text)
        domain = _primary_domain(domain_scores) or "general"

        # ----- Build unified result -----
        if extraction_result:
            # Use SystemExtractor results as the foundation
            system = extraction_result.system
            entities = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "properties": e.properties,
                }
                for e in system.entities
            ]
            relationships = [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.type,
                    "strength": r.strength,
                }
                for r in system.relationships
            ]
            constraints = [
                {
                    "id": c.id,
                    "name": c.name,
                    "type": c.type,
                    "value": c.value,
                }
                for c in system.constraints
            ]
            goals = []
            if system.goal:
                goals.append(system.goal)
            confidence = extraction_result.confidence
            missing_info = extraction_result.missing_info
        else:
            # Fallback: build from spaCy entities and keywords
            entities = []
            for i, ent in enumerate(spacy_entities):
                entities.append({
                    "id": f"entity_{i}",
                    "name": ent["text"],
                    "type": ent["label"].lower(),
                    "properties": {},
                })
            relationships = []
            constraints = []
            goals = []
            confidence = 0.3
            missing_info = ["entities", "constraints", "goal"]

            # Infer goal from problem type
            if problem_types:
                goals.append(problem_types[0][0].value.lower())

        # Enrich entities with spaCy NER results not already captured
        existing_names = {e["name"].lower() for e in entities}
        for ent in spacy_entities:
            if ent["text"].lower() not in existing_names:
                entities.append({
                    "id": f"spacy_{ent['label'].lower()}_{len(entities)}",
                    "name": ent["text"],
                    "type": ent["label"].lower(),
                    "properties": {},
                    "source": "spacy_ner",
                })
                existing_names.add(ent["text"].lower())

        return {
            "entities": entities,
            "relationships": relationships,
            "constraints": constraints,
            "goals": goals,
            "problem_type": primary_problem,
            "all_problem_types": [
                {"type": pt.value, "confidence": round(conf, 3),
                 "algorithm": PROBLEM_TO_ALGORITHM.get(pt, "")}
                for pt, conf in problem_types
            ],
            "domain": domain,
            "domain_scores": {k: round(v, 4) for k, v in domain_scores.items()},
            "confidence": confidence,
            "missing_info": missing_info,
        }
