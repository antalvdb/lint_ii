"""
Provider-agnostic LLM interface for LiNT-II.

Supports OpenAI, Anthropic, and Ollama providers with a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import logging
import os
import random
import re
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

    # Whether independent completion calls may run concurrently for one
    # document. True for remote HTTP APIs (many in-flight requests are fine and
    # cut wall-clock on long texts); False for on-machine providers (MLX: one
    # GPU/KV cache; Ollama: single local instance), where concurrent generation
    # thrashes rather than helps. Read by SuggestionEngine to size its job pool.
    supports_concurrency = False

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
        # Per-call token accounting, at INFO so it is present without turning on
        # DEBUG (which dumps full prompts and would pile tester text into the
        # logs). Counts and model name only -- no text, so this is safe to keep
        # on permanently. Every provider routes through this method, so it is
        # the one chokepoint that sees all traffic.
        #
        # Monthly totals:
        #   grep -oP 'LLM_USAGE .*\btotal=\K\d+' /var/log/lint-ii/app.log \
        #     | paste -sd+ | bc
        # Providers that return no usage block log total=? so the gap is
        # visible rather than silently counted as zero.
        if response.usage:
            logger.info(
                "LLM_USAGE model=%s prompt=%s completion=%s total=%s",
                response.model or self.model_name,
                response.usage.get("prompt_tokens", "?"),
                response.usage.get("completion_tokens", "?"),
                response.usage.get("total_tokens", "?"),
            )
        else:
            logger.info(
                "LLM_USAGE model=%s prompt=? completion=? total=?",
                response.model or self.model_name,
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
    supports_concurrency = True

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
    """Anthropic API (Claude), selected with LINT_PROVIDER=anthropic; the key
    comes from ANTHROPIC_API_KEY (set in the systemd EnvironmentFile, never in
    the repo). Like Mistral and Hetzner it sends tester text to an external
    service. Model via LINT_II_LLM_MODEL (or LINT_MODEL), default Claude Haiku
    5.5: about 40x cheaper than Opus 5.5 ($0.10/$0.50 per M tokens against
    $4/$20), and the job, short sentence rewrites, needs no more capability
    than the Qwen and Mistral models it is compared with. claude-sonnet-5-5 is
    the step up if Haiku falls short.

    Three differences from the other cloud providers matter for comparisons:

    - Thinking is always on for the current models and cannot be disabled, so
      its depth is set with an effort level instead (LINT_II_ANTHROPIC_EFFORT,
      default low: these are short, latency-bound rewrites). Thinking tokens
      count against max_tokens, which is why every call gets
      THINKING_HEADROOM extra tokens on top of what the caller asked for:
      without it the 512-token default is spent thinking and the answer comes
      back empty, the same failure Qwen had with thinking on. Output is billed
      as used, so the headroom costs nothing unless it is needed.
    - There is no temperature: the current models reject sampling parameters.
      Results are therefore not temperature-matched with Qwen (0.3) or Mistral
      (0.7) runs; say so when comparing.
    - A request the model declines ends with stop_reason "refusal". That is
      raised, and counted as a provider failure like any other error. On the
      models that support it (Opus, Sonnet; not Haiku) the request is first
      re-run on another model inside the same call ("fallbacks": "default"),
      and LLMResponse.model reports the model that actually answered.
      LINT_II_ANTHROPIC_FALLBACKS=0/1 overrides the per-model default.
    """

    DEFAULT_MODEL = "claude-haiku-5-5"
    THINKING_HEADROOM = 8000
    FALLBACK_BETA = "server-side-fallback-2026-07-01"
    supports_concurrency = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Pass api_key or set ANTHROPIC_API_KEY env var."
            )
        self._model = model or os.environ.get("LINT_II_LLM_MODEL", self.DEFAULT_MODEL)
        self._effort = os.environ.get("LINT_II_ANTHROPIC_EFFORT", "low")
        # Server-side fallback exists for Opus and Sonnet, not for Haiku.
        default_fallbacks = "0" if self._model.startswith("claude-haiku") else "1"
        self._fallbacks = os.environ.get("LINT_II_ANTHROPIC_FALLBACKS", default_fallbacks) != "0"
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
            # Same per-request ceiling as Hetzner; the SDK retries 429/5xx
            # with backoff, a little more patiently than its default of 2.
            self._client = Anthropic(api_key=self._api_key, timeout=120.0, max_retries=4)
        return self._client

    def _complete(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Generate completion using the Anthropic Messages API."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": (max_tokens or self.DEFAULT_MAX_TOKENS) + self.THINKING_HEADROOM,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self._effort},
            # The system prompts repeat across every call of an analysis, so
            # cache the prefix. A prefix below the model's minimum is simply
            # not cached, at no cost.
            "cache_control": {"type": "ephemeral"},
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        if self._fallbacks:
            response = client.beta.messages.create(
                betas=[self.FALLBACK_BETA], fallbacks="default", **kwargs
            )
        else:
            response = client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise RuntimeError(f"{response.model} declined the request (refusal, category={category})")
        if response.stop_reason == "max_tokens":
            logger.warning(
                "Anthropic response hit max_tokens (%s) on %s; the answer may be truncated",
                kwargs["max_tokens"], response.model,
            )

        u = response.usage
        prompt_tokens = (u.input_tokens + (u.cache_read_input_tokens or 0)
                         + (u.cache_creation_input_tokens or 0))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": u.output_tokens,  # includes thinking tokens
            "total_tokens": prompt_tokens + u.output_tokens,
        }

        # Thinking blocks come first (with empty text by default); the answer
        # is the text blocks.
        content = "".join(b.text for b in response.content if b.type == "text")

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
    supports_concurrency = True

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Mistral API key required. Pass api_key or set MISTRAL_API_KEY env var."
            )
        self._model = model or os.environ.get("LINT_II_LLM_MODEL", self.DEFAULT_MODEL)
        # 0.7 is the historical value every Mistral-era baseline in
        # scripts/eval/README.md was measured at, so it stays the default.
        # Overridable because Qwen moved to 0.3 after 0.7 produced noisy
        # lexical swaps: comparing Mistral@0.7 with Qwen@0.3 would confound
        # temperature with model. Mirrors LINT_II_HETZNER_TEMPERATURE.
        self._temperature = float(os.environ.get("LINT_II_MISTRAL_TEMPERATURE", "0.7"))
        # Reasoning models (mistral-large-4) think by default: the answer comes
        # back as a list of thinking + text blocks and the thinking eats the
        # max_tokens budget. "none" turns it off, matching Qwen's
        # enable_thinking:false. Only sent when set, because non-reasoning
        # models (mistral-large-2512) reject the parameter with a 400.
        self._reasoning_effort = os.environ.get("LINT_II_MISTRAL_REASONING_EFFORT") or None
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

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        usage = None
        if data.get("usage"):
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        content = data["choices"][0]["message"]["content"] or ""
        if isinstance(content, list):
            # Reasoning output: keep the text blocks, drop the thinking.
            content = "".join(
                block.get("text", "") for block in content if block.get("type") == "text"
            )

        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            usage=usage,
        )


# Provider answers worth retrying: rate limits and transient server errors.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_MAX_WAIT = 60.0


def _retry_wait(response: Any, attempt: int) -> float:
    """Seconds to wait before retry number `attempt` (0-based).

    Honours a numeric Retry-After header; otherwise backs off 5, 10, 20, 40s
    with jitter so the job threads of one analysis do not retry in lockstep."""
    header = response.headers.get("retry-after") if response is not None else None
    try:
        if header is not None:
            return min(max(float(header), 0.0), _RETRY_MAX_WAIT)
    except ValueError:
        pass  # HTTP-date form: fall back to the schedule
    return min(5.0 * 2 ** attempt, _RETRY_MAX_WAIT) * random.uniform(0.75, 1.25)


def _post_with_retries(
    client: Any, path: str, body: dict[str, Any], retries: int, provider: str
) -> Any:
    """POST, retrying 429/5xx answers up to `retries` times.

    Each retry logs one PROVIDER_RETRY warning. The eval runner counts the
    httpx line of every 429/5xx answer, so it subtracts these to count only
    the calls that finally failed (scripts/eval/run_eval.py). The last answer
    is raised as usual, so a call that never succeeds still counts as a
    provider failure. Connection errors are not retried here."""
    for attempt in range(retries + 1):
        response = client.post(path, json=body)
        if response.status_code not in _RETRY_STATUSES or attempt == retries:
            response.raise_for_status()
            return response
        wait = _retry_wait(response, attempt)
        logger.warning(
            "PROVIDER_RETRY provider=%s status=%d attempt=%d/%d wait=%.1fs",
            provider, response.status_code, attempt + 1, retries, wait,
        )
        time.sleep(wait)
    raise AssertionError("unreachable")


_THINK_BLOCK_RE = re.compile(r"(?s)\s*<think>.*?</think>\s*")


def _strip_think(content: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning from a completion.

    Qwen3-family models (the Hetzner endpoint serves Qwen3.6-35B-A3B) may prefix
    their answer with a reasoning block when 'thinking' mode is on. Our callers
    parse structured fields (BEHOUDEND/VOLLEDIG/HERSCHRIJVING) out of the raw
    text, so a stray reasoning block would break parsing. Strip well-formed
    blocks; if only a closing tag survives (some servers omit the opener), keep
    what follows the last one. A no-op when the model returns clean text."""
    cleaned = _THINK_BLOCK_RE.sub("", content)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    return cleaned.strip()


class HetznerProvider(LLMProvider):
    """Hetzner experimental inference API (inference.hetzner.com).

    OpenAI-compatible chat-completions surface (vLLM), selected with
    LINT_PROVIDER=hetzner; the key comes from the HETZNER_API_KEY environment
    variable (set in the systemd EnvironmentFile, never in the repo). A second
    cloud option alongside Mistral: sends tester text to an external service,
    with no local memory pressure or Metal wedge risk. The default model is the
    one the endpoint currently serves; override with LINT_MODEL when it rotates.

    The served Qwen3 model runs in 'thinking' mode by default: it emits a long
    reasoning trace (returned in a separate ``message.reasoning`` field) that
    burns the whole token budget before the actual answer, so on our structured
    prompts ``content`` comes back empty/truncated and nothing parses. We turn
    thinking off with the vLLM chat-template kwarg ``enable_thinking=False``.
    Set LINT_II_HETZNER_THINKING=1 to re-enable it (e.g. to compare quality)."""

    DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
    BASE_URL = "https://inference.hetzner.com/api/v1"
    supports_concurrency = True

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("HETZNER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Hetzner API key required. Pass api_key or set HETZNER_API_KEY env var."
            )
        self._model = model or os.environ.get("LINT_II_LLM_MODEL", self.DEFAULT_MODEL)
        self._thinking = os.environ.get("LINT_II_HETZNER_THINKING", "0") != "0"
        # Qwen3 at 0.7 (Mistral's tuned value) produces noisier lexical swaps —
        # occasional meaning/idiom errors on word_frequency replacements. A lower
        # temperature makes it more conservative; 0.3 is the default, overridable
        # via LINT_II_HETZNER_TEMPERATURE without a code change.
        self._temperature = float(os.environ.get("LINT_II_HETZNER_TEMPERATURE", "0.3"))
        # Since Oct 2026 the endpoint caps requests at ~20/min, below what one
        # long text needs: without retries a 429 silently costs suggestions.
        # 4 retries back off 5+10+20+40s (~75s), well inside the watchdog.
        self._retries = max(0, int(os.environ.get("LINT_II_HETZNER_RETRIES", "4")))
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
        """Generate completion using the Hetzner (OpenAI-compatible) API."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        if not self._thinking:
            # vLLM passes chat_template_kwargs into the Qwen chat template.
            body["chat_template_kwargs"] = {"enable_thinking": False}

        response = _post_with_retries(
            client, "/chat/completions", body, self._retries, "hetzner"
        )
        data = response.json()

        usage = None
        if data.get("usage"):
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        return LLMResponse(
            content=_strip_think(data["choices"][0]["message"]["content"] or ""),
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
        provider: Provider name ('openai', 'anthropic', 'ollama', 'mlx', 'mistral' or 'hetzner')
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
        "hetzner": HetznerProvider,
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
