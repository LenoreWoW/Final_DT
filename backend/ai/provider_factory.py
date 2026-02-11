"""
AI Provider Factory.

Selects the appropriate AI provider based on the AI_PROVIDER environment
variable. Defaults to 'local' (spaCy + rule-based NLP, no API key needed).

Supported values for AI_PROVIDER:
  - 'local' (default): spaCy NER + rule-based classification. Free, no keys.
  - 'anthropic': Claude API. Requires ANTHROPIC_API_KEY.

Any unrecognized value falls back to 'local' with a warning.
"""

import os
import logging

from .providers.base import AIProvider

logger = logging.getLogger(__name__)


def get_ai_provider() -> AIProvider:
    """
    Create and return the configured AI provider instance.

    Reads AI_PROVIDER from environment (default: 'local').
    """
    provider_name = os.getenv("AI_PROVIDER", "local").strip().lower()

    if provider_name == "local":
        from .providers.local import LocalAIProvider
        logger.info("Using LocalAIProvider (spaCy + rule-based NLP)")
        return LocalAIProvider()

    elif provider_name == "anthropic":
        from .providers.anthropic import AnthropicProvider
        logger.info("Using AnthropicProvider (Claude API)")
        return AnthropicProvider()

    else:
        logger.warning(
            "Unknown AI_PROVIDER '%s', falling back to local provider",
            provider_name,
        )
        from .providers.local import LocalAIProvider
        return LocalAIProvider()
