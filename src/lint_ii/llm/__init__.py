"""
LLM integration for LiNT-II suggestion generation.

This module provides a provider-agnostic interface for generating
readability improvement suggestions using large language models.

Supported providers:
- OpenAI (GPT-4o-mini default)
- Anthropic (Claude Haiku)
- Ollama (local models)

Example usage:
    from lint_ii import ReadabilityAnalysis

    analysis = ReadabilityAnalysis.from_text("Dutch text...")
    suggestions = analysis.generate_suggestions(
        llm_config={'provider': 'openai', 'api_key': '...'}
    )
"""

from lint_ii.llm.providers import (
    LLMProvider,
    LLMResponse,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    create_provider,
)
from lint_ii.llm.suggestions import (
    SuggestionEngine,
    SuggestionTrigger,
    Suggestion,
    SuggestionsResult,
    DEFAULT_THRESHOLDS,
)
from lint_ii.llm.prompts import PROMPT_TEMPLATES

__all__ = [
    # Providers
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "create_provider",
    # Suggestions
    "SuggestionEngine",
    "SuggestionTrigger",
    "Suggestion",
    "SuggestionsResult",
    "DEFAULT_THRESHOLDS",
    # Prompts
    "PROMPT_TEMPLATES",
]
