"""
Abstract base class for AI providers.

All AI providers (local NLP, Anthropic, OpenAI, etc.) must implement this
interface so the rest of the application can swap providers without code changes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class AIProvider(ABC):
    """
    Abstract AI provider interface.

    Every provider must support two core operations:
      1. chat() - Process a single user message in context of conversation history.
      2. extract_system() - Analyze a full conversation to extract a structured
         system definition suitable for digital twin generation.
    """

    @abstractmethod
    async def chat(self, message: str, conversation_history: list) -> dict:
        """
        Process a user message and return an AI response.

        Args:
            message: The latest user message.
            conversation_history: List of prior messages, each a dict with
                keys 'role' ('user'|'assistant') and 'content' (str).

        Returns:
            dict with keys:
                - response (str): The text reply to show the user.
                - extracted_entities (list): Entities found in this message.
                - problem_type (str|None): Detected problem category, e.g.
                  'OPTIMIZATION', 'CLASSIFICATION', 'SIMULATION', etc.
                - domain (str|None): Detected domain, e.g. 'healthcare'.
                - conversation_state (str): Current state in the conversation
                  flow (e.g. 'greeting', 'problem_description').
                - follow_up_questions (list[str]): Suggested follow-up questions.
        """
        pass

    @abstractmethod
    async def extract_system(self, conversation: list) -> dict:
        """
        Analyze a full conversation and extract a structured system definition.

        Args:
            conversation: Complete conversation history (list of dicts with
                'role' and 'content').

        Returns:
            dict with keys:
                - entities (list): System entities with properties.
                - relationships (list): Connections between entities.
                - constraints (list): System constraints.
                - goals (list): User objectives.
                - problem_type (str): Primary problem classification.
                - domain (str): Detected application domain.
                - confidence (float): Extraction confidence 0.0-1.0.
                - missing_info (list[str]): Information still needed.
        """
        pass
