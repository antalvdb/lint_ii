"""HetznerProvider retries rate-limited and transient-error answers.

The endpoint caps requests at ~20/min since Oct 2026; without retries a 429
silently costs suggestions. No test makes a real API call or really sleeps.
"""

import httpx
import pytest

from lint_ii.llm import providers
from lint_ii.llm.providers import HetznerProvider

_OK = {
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "choices": [{"message": {"content": "Herschreven zin."}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _response(status, headers=None):
    request = httpx.Request("POST", "https://inference.hetzner.com/api/v1/chat/completions")
    return httpx.Response(status, json=_OK if status == 200 else {}, headers=headers or {},
                          request=request)


class _Client:
    def __init__(self, *statuses):
        self.answers = [s if isinstance(s, httpx.Response) else _response(s) for s in statuses]
        self.calls = 0

    def post(self, path, json):
        self.calls += 1
        return self.answers.pop(0)


@pytest.fixture
def waits(monkeypatch):
    slept = []
    monkeypatch.setattr(providers.time, "sleep", slept.append)
    return slept


def _provider(monkeypatch, client, **env):
    monkeypatch.setenv("HETZNER_API_KEY", "test-key")
    monkeypatch.delenv("LINT_II_HETZNER_RETRIES", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    provider = HetznerProvider()
    provider._client = client
    return provider


def test_a_429_is_retried_until_it_succeeds(monkeypatch, waits, caplog):
    client = _Client(429, 429, 200)
    out = _provider(monkeypatch, client)._complete("prompt")
    assert out.content == "Herschreven zin."
    assert client.calls == 3
    assert len(waits) == 2 and waits[0] < waits[1]
    assert caplog.text.count("PROVIDER_RETRY provider=hetzner status=429") == 2


def test_retry_after_header_is_honoured_and_capped(monkeypatch, waits):
    client = _Client(_response(429, {"Retry-After": "7"}), _response(503, {"Retry-After": "900"}), 200)
    _provider(monkeypatch, client)._complete("prompt")
    assert waits == [7.0, providers._RETRY_MAX_WAIT]


def test_gives_up_after_the_configured_retries(monkeypatch, waits):
    client = _Client(429, 429, 429)
    with pytest.raises(httpx.HTTPStatusError, match="429"):
        _provider(monkeypatch, client, LINT_II_HETZNER_RETRIES="2")._complete("prompt")
    assert client.calls == 3


def test_client_errors_are_not_retried(monkeypatch, waits):
    client = _Client(422)
    with pytest.raises(httpx.HTTPStatusError, match="422"):
        _provider(monkeypatch, client)._complete("prompt")
    assert client.calls == 1 and waits == []


def test_retries_can_be_turned_off(monkeypatch, waits):
    client = _Client(429)
    with pytest.raises(httpx.HTTPStatusError):
        _provider(monkeypatch, client, LINT_II_HETZNER_RETRIES="0")._complete("prompt")
    assert client.calls == 1
