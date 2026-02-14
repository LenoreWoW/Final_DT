"""Conversational quantum AI stub."""

import enum, uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class UserExpertise(enum.Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


@dataclass
class ConversationResponse:
    message: str = ""
    requires_input: bool = True
    options: List[str] = field(default_factory=list)


class ConversationalQuantumAI:
    def __init__(self):
        self._sessions = {}

    async def start_conversation(self, user_id: str) -> Tuple[str, ConversationResponse]:
        session_id = str(uuid.uuid4())[:12]
        self._sessions[session_id] = {"user_id": user_id, "history": []}
        return session_id, ConversationResponse(
            message="Welcome to the Quantum Digital Twin Platform! How can I help you today?",
            requires_input=True,
            options=["Create a quantum twin", "Analyze data", "Run simulation"],
        )

    async def continue_conversation(self, session_id: str, message: str) -> ConversationResponse:
        session = self._sessions.get(session_id, {"history": []})
        session["history"].append(message)
        return ConversationResponse(
            message=f"I understand you said: '{message[:50]}...'. Let me help with that.",
            requires_input=True,
        )
