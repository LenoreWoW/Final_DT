"""Healthcare conversational AI stub."""

import enum
from dataclasses import dataclass
from typing import Dict, Any


class UserRole(enum.Enum):
    PATIENT = "patient"
    PHYSICIAN = "physician"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class HealthcareConversationalAI:
    def __init__(self, config=None):
        self._config = config or {}

    async def chat(self, message, role=UserRole.PHYSICIAN):
        return {
            "response": f"Healthcare AI response to: {message[:50]}",
            "intent": "query",
            "confidence": 0.85,
        }
