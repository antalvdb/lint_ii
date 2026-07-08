"""
Provider-agnostic LLM interface for LiNT-II.

Supports OpenAI, Anthropic, and Ollama providers with a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Watchdog ceiling for a single completion call, in seconds. Normal calls take
# 5-60s; the whole-document spelling pass on a long text can reach a few
# minutes. The failure mode this guards against is unbounded: an MLX generate
# call can hang forever inside the Metal driver (observed 2026-06-10) while the
# process stays healthy, leaving jobs pending indefinitely with no error.
LLM_TIMEOUT_SECONDS = float(os.environ.get("LINT_II_LLM_TIMEOUT", "300"))

# Set to a timestamp when a completion call times out. While set, further calls
# fail fast instead of each waiting out the full timeout — a wedged Metal queue
# does not recover without a process restart (or the stuck call finishing after
# all, which clears the flag). Read by the API health endpoint.
_wedged_at: float | None = None
_wedged_lock = threading.Lock()


def llm_wedged_since() -> float | None:
    """Timestamp of the first unresolved completion timeout, or None if healthy."""
    return _wedged_at


def _mark_wedged() -> None:
    global _wedged_at
    with _wedged_lock:
        if _wedged_at is None:
            _wedged_at = time.time()


def _clear_wedged() -> None:
    global _wedged_at
    with _wedged_lock:
        _wedged_at = None


class LLMTimeoutError(RuntimeError):
    """A completion call exceeded LLM_TIMEOUT_SECONDS (or the provider is wedged)."""


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: dict[str, int] | None = None  # tokens used


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    # Default output ceiling. A single rewritten sentence plus a short
    # explanation fits comfortably; the whole-document spelling pass overrides
    # this with a larger value since it enumerates every error in one call.
    DEFAULT_MAX_TOKENS = 512

    @abstractmethod
    def _complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        pass

    def complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        logger.debug(
            "LLM PROMPT [%s]\n--- system ---\n%s\n--- user ---\n%s\n--- end ---",
            self.model_name,
            system_prompt or "(none)",
            prompt,
        )
        response = self._complete_with_watchdog(prompt, system_prompt, max_tokens)
        logger.debug(
            "LLM RESPONSE [%s]\n%s\n--- end ---",
            self.model_name,
            response.content,
        )
        return response

    def _complete_with_watchdog(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Run _complete on a watchdog thread so a wedged GPU driver surfaces as
        an error instead of a forever-pending job. The stuck thread cannot be
        interrupted (it is inside an iokit trap), so it is left behind as a
        daemon thread; while the wedge lasts, further calls fail fast."""
        if LLM_TIMEOUT_SECONDS <= 0:
            return self._complete(prompt, system_prompt, max_tokens)

        if _wedged_at is not None:
            raise LLMTimeoutError(
                f"LLM provider marked wedged since {time.strftime('%H:%M:%S', time.localtime(_wedged_at))}; "
                "failing fast (restart the server to recover)"
            )

        outcome: dict[str, Any] = {}
        timed_out = threading.Event()

        def _run() -> None:
            try:
                outcome["response"] = self._complete(prompt, system_prompt, max_tokens)
            except BaseException as e:
                outcome["error"] = e
            finally:
                if timed_out.is_set():
                    # The call finished after the watchdog gave up on it: the
                    # model is slow, not wedged. Let new calls through again.
                    logger.warning(
                        "LLM call completed %s after its watchdog timeout fired; "
                        "clearing wedged state", self.model_name,
                    )
                    _clear_wedged()

        worker = threading.Thread(target=_run, name="llm-complete", daemon=True)
        worker.start()
        worker.join(LLM_TIMEOUT_SECONDS)

        if worker.is_alive():
            timed_out.set()
            _mark_wedged()
            logger.critical(
                "LLM call did not finish within %.0fs — provider likely wedged "
                "(Metal driver hang). Marking degraded; restart the server to recover.",
                LLM_TIMEOUT_SECONDS,
            )
            raise LLMTimeoutError(
                f"LLM call exceeded {LLM_TIMEOUT_SECONDS:.0f}s watchdog timeout"
            )

        if "error" in outcome:
            raise outcome["error"]
        return outcome["response"]

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

    def _complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
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
            max_tokens=max_tokens or self.DEFAULT_MAX_TOKENS,
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

    def _complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
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

    def _complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
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
                "options": {"num_predict": max_tokens or self.DEFAULT_MAX_TOKENS},
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self._model,
            usage=None,  # Ollama doesn't provide token counts in the same format
        )


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class MistralProvider(LLMProvider):
    """Mistral API provider (api.mistral.ai).

    Cloud alternative to the on-machine MLX provider: no local memory
    pressure and no Metal wedge risk, at the cost of sending tester text to
    an external service. Selected with LINT_PROVIDER=mistral; the key comes
    from the MISTRAL_API_KEY environment variable (set in the launchd plist,
    never in the repo)."""

    DEFAULT_MODEL = "mistral-large-latest"
    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Mistral API key required. Pass api_key or set MISTRAL_API_KEY env var."
            )
        self._model = model or os.environ.get("LINT_II_LLM_MODEL", self.DEFAULT_MODEL)
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx package not installed. Install with: pip install lint_ii[llm]"
                )
            self._client = httpx.Client(
                base_url=self.BASE_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=120.0,
            )
        return self._client

    def _complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Generate completion using the Mistral chat completions API."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            },
        )
        response.raise_for_status()
        data = response.json()

        usage = None
        if data.get("usage"):
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        return LLMResponse(
            content=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self._model),
            usage=usage,
        )


class MLXProvider(LLMProvider):
    """Apple Silicon MLX provider — loads model directly for fast on-device inference."""

    DEFAULT_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"

    # Ceiling on prompt tokens retained in the persistent KV cache between
    # calls. KV state is ~256 KB per token for the 32B model, so 1024 tokens
    # pins ~260 MB — enough to hold any full system prompt, while a
    # whole-document spelling prompt would otherwise pin gigabytes.
    PROMPT_CACHE_MAX_TOKENS = 1024

    def __init__(self, model: str | None = None):
        self._model_path = model or os.environ.get("MLX_MODEL", self.DEFAULT_MODEL)
        self._model = None
        self._tokenizer = None
        # All rewrite prompts within one analysis share the chat-template
        # header + system prompt; keeping their KV state warm skips re-evaluating
        # those tokens on every call. Guarded by a lock because completion calls
        # run on watchdog threads (serialized in practice by the single-worker
        # analysis executor).
        self._prompt_cache = None
        self._prompt_cache_tokens: list[int] = []
        self._generate_lock = threading.Lock()
        self._prompt_cache_enabled = os.environ.get("LINT_II_MLX_PROMPT_CACHE", "1") != "0"

    @property
    def model_name(self) -> str:
        return self._model_path

    def load(self) -> None:
        """Load model into memory. Call once at startup to avoid first-request delay."""
        if self._model is not None:
            return
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "mlx-lm not installed. Install with: pip install mlx-lm"
            )
        self._model, self._tokenizer = load(self._model_path)

    def _complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Generate completion using MLX on Apple Silicon."""
        from mlx_lm import generate
        self.load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokens = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        # Per-trigger rewrites fit in the 512-token default; the spelling pass
        # passes a larger ceiling. The model stops at EOS in the common case, so
        # this only caps runaway generation rather than slowing normal calls.
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        if self._prompt_cache_enabled:
            with self._generate_lock:
                content = self._generate_with_prompt_cache(tokens, max_tokens)
        else:
            content = generate(
                self._model,
                self._tokenizer,
                prompt=tokens,
                max_tokens=max_tokens,
                verbose=False,
            )

        return LLMResponse(
            content=content,
            model=self._model_path,
            usage=None,
        )

    def _generate_with_prompt_cache(self, tokens: list[int], max_tokens: int) -> str:
        """Generate while reusing the KV state of the longest token prefix
        shared with the previous call (typically the chat-template header plus
        system prompt). Caller must hold _generate_lock."""
        from mlx_lm import generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        # _prompt_cache_tokens mirrors exactly what the KV cache holds; if the
        # two ever disagree (e.g. an aborted generation), rebuild from scratch.
        if self._prompt_cache is None or (
            self._prompt_cache_tokens
            and self._prompt_cache[0].offset != len(self._prompt_cache_tokens)
        ):
            self._prompt_cache = make_prompt_cache(self._model)
            self._prompt_cache_tokens = []

        # generate() must be fed at least one token, so never reuse the full prompt.
        common = min(
            _common_prefix_len(self._prompt_cache_tokens, tokens), len(tokens) - 1
        )
        excess = len(self._prompt_cache_tokens) - common
        if excess > 0 and trim_prompt_cache(self._prompt_cache, excess) != excess:
            self._prompt_cache = make_prompt_cache(self._model)
            self._prompt_cache_tokens = []
            common = 0
        else:
            self._prompt_cache_tokens = self._prompt_cache_tokens[:common]

        logger.info(
            "MLX prompt cache: reusing %d/%d prompt tokens", common, len(tokens)
        )

        try:
            content = generate(
                self._model,
                self._tokenizer,
                prompt=tokens[common:],
                max_tokens=max_tokens,
                verbose=False,
                prompt_cache=self._prompt_cache,
            )
        except BaseException:
            # The cache may hold a partially prefilled prompt; discard it.
            self._prompt_cache = None
            self._prompt_cache_tokens = []
            raise

        # The cache now also holds this call's suffix and generated tokens.
        # Trim back to the (capped) prompt so the next call can reuse its
        # shared prefix without unbounded KV growth.
        keep = min(len(tokens), self.PROMPT_CACHE_MAX_TOKENS)
        surplus = self._prompt_cache[0].offset - keep
        if surplus > 0 and trim_prompt_cache(self._prompt_cache, surplus) != surplus:
            self._prompt_cache = None
            self._prompt_cache_tokens = []
            return content
        self._prompt_cache_tokens = tokens[:keep]
        return content


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
        "mlx": MLXProvider,
        "mistral": MistralProvider,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Choose from: {', '.join(providers.keys())}"
        )

    provider_class = providers[provider]

    if provider in ("ollama", "mlx"):
        return provider_class(model=model, **kwargs)
    else:
        return provider_class(api_key=api_key, model=model, **kwargs)
