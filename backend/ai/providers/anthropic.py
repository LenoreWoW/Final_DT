"""
Anthropic (Claude) AI Provider -- requires ANTHROPIC_API_KEY.

This is a placeholder for future Claude API integration. It validates
that the API key is set at construction time and returns placeholder
responses directing users to the free local provider.
"""

import os
import logging
from .base import AIProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(AIProvider):
    """
    AI provider backed by Anthropic's Claude API.

    Requires the ANTHROPIC_API_KEY environment variable to be set.
    If unavailable, use AI_PROVIDER=local for the free spaCy-based provider.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set AI_PROVIDER=local in your .env to use the free "
                "local NLP provider (spaCy + rule-based), or provide "
                "a valid Anthropic API key."
            )

    async def chat(self, message: str, conversation_history: list) -> dict:
        """Send a chat message to Claude API. Not yet implemented."""
        logger.warning("Anthropic chat() called but not yet implemented; returning placeholder")
        return {
            "message": (
                "The Anthropic provider is not yet fully implemented. "
                "Please set AI_PROVIDER=local in your .env to use the "
                "working local NLP provider."
            ),
            "confidence": 0.0,
            "provider": "anthropic",
            "status": "not_implemented",
        }

    async def extract_system(self, conversation: list) -> dict:
        """Extract system definition via Claude API. Not yet implemented."""
        logger.warning("Anthropic extract_system() called but not yet implemented; returning placeholder")
        return {
            "entities": [],
            "relationships": [],
            "domain": "unknown",
            "confidence": 0.0,
            "provider": "anthropic",
            "status": "not_implemented",
        }
