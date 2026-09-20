"""Shared endpoint config for the probes.

The probes talk to the raw chat-completions surface on purpose (no provider
wrapper, no watchdog) so they measure the model, not the pipeline. But the URL
used to be hardcoded to Hetzner in all four, which made "does another provider
emit parseable blocks?" unanswerable without editing code -- the first question
worth asking in any provider migration.

Override with env vars, Hetzner remaining the default:

    LINT_II_PROBE_BASE_URL   default https://inference.hetzner.com/api/v1
    LINT_II_PROBE_API_KEY    default the value of $HETZNER_API_KEY
    LINT_II_PROBE_AUTH       "bearer" (default) or "header:X-Api-Key"
    LINT_II_LLM_MODEL        default Qwen/Qwen3.6-35B-A3B-FP8
    LINT_II_PROBE_NO_THINK   "0" to omit chat_template_kwargs entirely

Example -- point the probes at another OpenAI-compatible endpoint:

    LINT_II_PROBE_BASE_URL=https://example.uu.nl/v1 \
    LINT_II_PROBE_API_KEY="$UU_API_KEY" \
    LINT_II_LLM_MODEL=mistral-small \
    python3 connective_probe.py --reps 6 --workers 2
"""
from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://inference.hetzner.com/api/v1"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


def model() -> str:
    return os.environ.get("LINT_II_LLM_MODEL", DEFAULT_MODEL)


def base_url() -> str:
    return os.environ.get("LINT_II_PROBE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def headers() -> dict[str, str]:
    key = os.environ.get("LINT_II_PROBE_API_KEY") or os.environ.get("HETZNER_API_KEY")
    if not key:
        raise SystemExit(
            "No API key. Set LINT_II_PROBE_API_KEY, or HETZNER_API_KEY for the "
            "default endpoint (box: /etc/lint-ii/lint-ii.env)."
        )
    auth = os.environ.get("LINT_II_PROBE_AUTH", "bearer")
    if auth.startswith("header:"):
        return {auth.split(":", 1)[1]: key, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def body(messages: list[dict], max_tokens: int, temperature: float | None = None) -> dict:
    b = {
        "model": model(),
        "messages": messages,
        "temperature": float(
            os.environ.get("LINT_II_HETZNER_TEMPERATURE", "0.3")
            if temperature is None else temperature
        ),
        "max_tokens": max_tokens,
    }
    # Qwen-specific: thinking defaults ON and burns the budget before the answer.
    # Harmless on servers that ignore unknown chat_template_kwargs, but set
    # LINT_II_PROBE_NO_THINK=0 for any server that rejects them outright.
    if os.environ.get("LINT_II_PROBE_NO_THINK", "1") != "0":
        b["chat_template_kwargs"] = {"enable_thinking": False}
    return b


def describe() -> str:
    return f"{base_url()}  model={model()}  auth={os.environ.get('LINT_II_PROBE_AUTH','bearer')}"
