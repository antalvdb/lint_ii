"""
Provider-agnostic LLM interface for LiNT-II.

Supports OpenAI, Anthropic, and Ollama providers with a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import os


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: dict[str, int] | None = None  # tokens used


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt to complete
            system_prompt: Optional system prompt for context

        Returns:
            LLMResponse with the generated content
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4o-mini default)."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (defaults to gpt-4o-mini or LINT_II_LLM_MODEL env var)
            base_url: Optional custom base URL for API-compatible services
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key or set OPENAI_API_KEY env var."
            )

        self._model = model or os.environ.get("LINT_II_LLM_MODEL", self.DEFAULT_MODEL)
        self._base_url = base_url
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install lint_ii[llm]"
                )
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate completion using OpenAI API."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
        )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage=usage,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude Haiku default)."""

    DEFAULT_MODEL = "claude-3-5-haiku-latest"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (defaults to claude-3-5-haiku-latest)
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Pass api_key or set ANTHROPIC_API_KEY env var."
            )

        self._model = model or self.DEFAULT_MODEL
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install lint_ii[llm]"
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate completion using Anthropic API."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
        )


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model to use (defaults to llama3.2)
            base_url: Ollama API URL (defaults to http://localhost:11434)
        """
        self._model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        """Lazy-load the httpx client."""
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx package not installed. Install with: pip install lint_ii[llm]"
                )
            self._client = httpx.Client(base_url=self._base_url, timeout=600.0)
        return self._client

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate completion using Ollama API."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.post(
            "/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self._model,
            usage=None,  # Ollama doesn't provide token counts in the same format
        )


def create_provider(
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider: Provider name ('openai', 'anthropic', or 'ollama')
        api_key: API key for the provider (not needed for ollama)
        model: Model to use (provider-specific defaults if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLMProvider instance

    Example:
        >>> provider = create_provider('openai', api_key='sk-...')
        >>> provider = create_provider('anthropic')  # uses env var
        >>> provider = create_provider('ollama', model='mistral')
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Choose from: {', '.join(providers.keys())}"
        )

    provider_class = providers[provider]

    if provider == "ollama":
        return provider_class(model=model, **kwargs)
    else:
        return provider_class(api_key=api_key, model=model, **kwargs)
