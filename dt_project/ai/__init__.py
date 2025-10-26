"""
🤖 AI-Powered Quantum Digital Twin Creation
===========================================

Conversational AI and intelligent mapping for automated quantum digital twin creation.

Features:
- Natural language understanding
- Intelligent quantum advantage mapping
- Healthcare-specialized conversational flows
- Data-to-quantum-approach recommendation

Author: Hassan Al-Sahli
Purpose: Democratize quantum computing through AI
Reference: docs/CONVERSATIONAL_AI_INTEGRATION_PLAN.md
"""

from .conversational_quantum_ai import (
    ConversationalQuantumAI,
    ConversationState,
    ConversationContext,
    UserExpertise
)

from .intelligent_quantum_mapper import (
    IntelligentQuantumMapper,
    QuantumAdvantageMapping,
    IntelligentMappingResult
)

from .healthcare_conversational_ai import (
    HealthcareConversationalAI,
    HealthcareTwinResult,
    ClinicalInterpretation
)

__all__ = [
    'ConversationalQuantumAI',
    'ConversationState',
    'ConversationContext',
    'UserExpertise',
    'IntelligentQuantumMapper',
    'QuantumAdvantageMapping',
    'IntelligentMappingResult',
    'HealthcareConversationalAI',
    'HealthcareTwinResult',
    'ClinicalInterpretation',
]

__version__ = '1.0.0'
__author__ = 'Hassan Al-Sahli'
