"""MistralProvider request and response handling, with a fake HTTP client.

mistral-large-4 is a reasoning model: by default its message content is a
list of thinking and text blocks, and reasoning is switched off with
reasoning_effort "none". mistral-large-2512 returns a plain string and
rejects reasoning_effort with a 400. No test makes a real API call.
"""

from types import SimpleNamespace

from lint_ii.llm.providers import MistralProvider


class _Client:
    def __init__(self, content):
        self.content = content
        self.payload = None

    def post(self, path, json):
        self.payload = json
        body = {
            "model": json["model"],
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)


def _provider(monkeypatch, content="Herschreven zin.", **env):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    for name in ("LINT_II_LLM_MODEL", "LINT_II_MISTRAL_TEMPERATURE",
                 "LINT_II_MISTRAL_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    provider = MistralProvider()
    provider._client = _Client(content)
    return provider, provider._client


def test_plain_string_content(monkeypatch):
    provider, client = _provider(monkeypatch)
    out = provider._complete("prompt")
    assert out.content == "Herschreven zin."
    assert out.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_reasoning_effort_is_not_sent_by_default(monkeypatch):
    # mistral-large-2512 answers 400 "reasoning_effort is not enabled".
    provider, client = _provider(monkeypatch)
    provider._complete("prompt")
    assert "reasoning_effort" not in client.payload
    assert client.payload["temperature"] == 0.7


def test_reasoning_effort_and_temperature_come_from_the_environment(monkeypatch):
    provider, client = _provider(monkeypatch, LINT_II_LLM_MODEL="mistral-large-4",
                                 LINT_II_MISTRAL_REASONING_EFFORT="none",
                                 LINT_II_MISTRAL_TEMPERATURE="0.3")
    provider._complete("prompt")
    assert client.payload["model"] == "mistral-large-4"
    assert client.payload["reasoning_effort"] == "none"
    assert client.payload["temperature"] == 0.3


def test_reasoning_blocks_keep_only_the_text(monkeypatch):
    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "De gebruiker vraagt..."}]},
        {"type": "text", "text": "De gemeente wil het aanvragen makkelijker maken."},
    ]
    provider, _ = _provider(monkeypatch, content=content)
    assert provider._complete("prompt").content == "De gemeente wil het aanvragen makkelijker maken."
