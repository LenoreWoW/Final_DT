"""
⚛️🤖 Quantum-Powered AI
=======================

Revolutionary AI-Powered Quantum Digital Twin Creation with TRUE Quantum Intelligence.

Features:
- ⚛️ Quantum Natural Language Processing (QNLP)
- 🎯 Quantum Intent Classification (Quantum ML)
- 💬 Quantum Semantic Understanding (Quantum Embeddings)
- 🔮 Quantum Response Generation (Quantum Creativity)
- 🧬 Quantum Conversation Memory (Entangled Context)
- 🌍 Universal Domain Support (Healthcare, Finance, Manufacturing, Energy, etc.)
- 🔨 Dynamic Quantum Twin Creation (Any Subject)

Author: Hassan Al-Sahli
Purpose: Democratize quantum computing through quantum-powered AI
"""

import sys
import logging

logger = logging.getLogger(__name__)

# New Quantum-Powered AI
try:
    from .quantum_conversational_ai import (
        QuantumConversationalAI,
        quantum_ai,
        chat_with_quantum_ai,
        QuantumWordEmbedding,
        QuantumIntentClassifier,
        QuantumSemanticUnderstanding,
        QuantumResponseGenerator,
        QuantumConversationMemory
    )
    QUANTUM_AI_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Quantum AI not available: {e}")
    QUANTUM_AI_AVAILABLE = False
    quantum_ai = None
    chat_with_quantum_ai = None

try:
    from .universal_ai_interface import (
        UniversalQuantumAI,
        universal_ai,
        ask_quantum_ai,
        UniversalDomain
    )
    UNIVERSAL_AI_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Universal AI not available: {e}")
    UNIVERSAL_AI_AVAILABLE = False
    universal_ai = None
    ask_quantum_ai = None

# Domain mapping and consultation (for backwards compatibility)
try:
    from .quantum_domain_mapper import (
        IntelligentQuantumMapper,
        QuantumAdvantageMapping,
        IntelligentMappingResult
    )
    MAPPER_AVAILABLE = True
except ImportError:
    MAPPER_AVAILABLE = False

try:
    from .quantum_twin_consultant import (
        ConversationalQuantumAI,
        ConversationContext,
        ConversationResponse
    )
    CONSULTANT_AVAILABLE = True
except ImportError:
    CONSULTANT_AVAILABLE = False

try:
    from .healthcare_conversational_ai import (
        HealthcareConversationalAI,
        HealthcareTwinResult,
        ClinicalInterpretation
    )
    HEALTHCARE_AI_AVAILABLE = True
except ImportError:
    HEALTHCARE_AI_AVAILABLE = False

# =============================================================================
# BACKWARD COMPATIBILITY: Create module-level aliases
# These allow imports like: from dt_project.ai.universal_conversational_ai import ...
# =============================================================================
try:
    from . import universal_ai_interface
    sys.modules['dt_project.ai.universal_conversational_ai'] = universal_ai_interface
except (ImportError, AttributeError):
    pass

try:
    from . import quantum_conversational_ai
    # Also alias any old names that tests might use
except (ImportError, AttributeError):
    pass

__all__ = [
    # New Quantum AI (Primary)
    'QuantumConversationalAI',
    'quantum_ai',
    'chat_with_quantum_ai',
    'QuantumWordEmbedding',
    'QuantumIntentClassifier',
    'QuantumSemanticUnderstanding',
    'QuantumResponseGenerator',
    'QuantumConversationMemory',
    'UniversalQuantumAI',
    'universal_ai',
    'ask_quantum_ai',
    'UniversalDomain',

    # Old modules (if available)
    'IntelligentQuantumMapper',
    'QuantumAdvantageMapping',
    'IntelligentMappingResult',
    'HealthcareConversationalAI',
    'HealthcareTwinResult',
    'ClinicalInterpretation',
    'ConversationalQuantumAI',
    'ConversationContext',
    'ConversationResponse',
]

__version__ = '2.1.0'  # Quantum AI version with backward compatibility
__author__ = 'Hassan Al-Sahli'
