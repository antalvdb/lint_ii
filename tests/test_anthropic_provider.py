"""AnthropicProvider against the current Claude models, with a fake client.

The current models always think: thinking blocks come before the answer and
count against max_tokens, sampling parameters are rejected, and a declined
request ends with stop_reason "refusal". These tests pin how the provider
handles each of those. No test makes a real API call.
"""

from types import SimpleNamespace

import pytest

from lint_ii.llm.providers import AnthropicProvider, LLMProvider


def _response(stop_reason="end_turn", model="claude-haiku-5-5", text="Herschreven zin."):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        model=model,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=100, output_tokens=40,
                              cache_read_input_tokens=900, cache_creation_input_tokens=0),
    )


class _Endpoint:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _provider(monkeypatch, response=None, **env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for name in ("LINT_II_LLM_MODEL", "LINT_II_ANTHROPIC_EFFORT", "LINT_II_ANTHROPIC_FALLBACKS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    provider = AnthropicProvider()
    plain, beta = _Endpoint(response or _response()), _Endpoint(response or _response())
    provider._client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    return provider, plain, beta


def test_answer_is_the_text_after_the_thinking_block(monkeypatch):
    provider, plain, _ = _provider(monkeypatch)
    out = provider._complete("prompt", system_prompt="system")
    assert out.content == "Herschreven zin."
    assert plain.kwargs["system"] == "system"


def test_thinking_gets_headroom_and_no_sampling_parameters_are_sent(monkeypatch):
    provider, plain, _ = _provider(monkeypatch)
    assert provider.model_name == "claude-haiku-5-5"
    provider._complete("prompt", max_tokens=300)
    assert plain.kwargs["max_tokens"] == 300 + AnthropicProvider.THINKING_HEADROOM
    provider._complete("prompt")
    assert plain.kwargs["max_tokens"] == LLMProvider.DEFAULT_MAX_TOKENS + AnthropicProvider.THINKING_HEADROOM
    assert plain.kwargs["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(plain.kwargs)


def test_model_and_effort_come_from_the_environment(monkeypatch):
    provider, _, beta = _provider(monkeypatch, LINT_II_LLM_MODEL="claude-sonnet-5-5",
                                  LINT_II_ANTHROPIC_EFFORT="medium")  # Sonnet: fallbacks on
    provider._complete("prompt")
    assert provider.model_name == "claude-sonnet-5-5"
    assert beta.kwargs["model"] == "claude-sonnet-5-5"
    assert beta.kwargs["output_config"] == {"effort": "medium"}


def test_haiku_has_no_fallbacks_by_default(monkeypatch):
    # Server-side fallback does not exist for Haiku; sending it would 400.
    provider, plain, beta = _provider(monkeypatch)
    provider._complete("prompt")
    assert beta.kwargs is None
    assert "fallbacks" not in plain.kwargs and "betas" not in plain.kwargs


def test_fallbacks_are_on_by_default_for_opus_and_report_the_serving_model(monkeypatch):
    provider, plain, beta = _provider(monkeypatch, response=_response(model="claude-opus-4-8"),
                                      LINT_II_LLM_MODEL="claude-opus-5-5")
    out = provider._complete("prompt")
    assert beta.kwargs["fallbacks"] == "default"
    assert beta.kwargs["betas"] == [AnthropicProvider.FALLBACK_BETA]
    assert plain.kwargs is None
    assert out.model == "claude-opus-4-8"


def test_fallbacks_can_be_turned_off(monkeypatch):
    provider, plain, beta = _provider(monkeypatch, LINT_II_LLM_MODEL="claude-sonnet-5-5",
                                      LINT_II_ANTHROPIC_FALLBACKS="0")
    provider._complete("prompt")
    assert beta.kwargs is None
    assert "fallbacks" not in plain.kwargs and "betas" not in plain.kwargs


def test_a_refusal_is_an_error_not_an_empty_answer(monkeypatch):
    # An empty answer would silently cost the suggestion; an error is counted
    # as a provider failure, so the incomplete result is reported and never
    # cached.
    provider, _, _ = _provider(monkeypatch, response=_response(stop_reason="refusal", text=""))
    with pytest.raises(RuntimeError, match="refusal"):
        provider._complete("prompt")


def test_usage_counts_cached_prompt_tokens(monkeypatch):
    provider, _, _ = _provider(monkeypatch)
    usage = provider._complete("prompt").usage
    assert usage == {"prompt_tokens": 1000, "completion_tokens": 40, "total_tokens": 1040}
