"""POST /sentence-metrics: scores text the user wrote (an edited suggestion).

Needs the [server] extras (fastapi, langid); skipped where they are absent.
The TestClient is used without a context manager, so the app's lifespan
(example-text warm-up, LLM provider start-up) never runs: no model is called.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langid")
os.environ.setdefault("LINT_PROVIDER", "ollama")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

client = TestClient(api.app)


def test_returns_suggestion_shaped_metrics():
    r = client.post(
        "/sentence-metrics",
        json={"text": "De aannemer begint na de zomer. Dat heeft de gemeente besloten."},
    )
    assert r.status_code == 200
    m = r.json()
    assert m["n_sentences"] == 2
    assert len(m["sdl_values"]) == 2
    for key in ("word_freq_sum", "word_freq_count", "cwpc_values",
                "n_concrete", "n_abstract", "n_undefined"):
        assert key in m


@pytest.mark.parametrize("text", ["", "   \n "])
def test_empty_text_is_rejected(text):
    assert client.post("/sentence-metrics", json={"text": text}).status_code == 422


def test_oversized_text_is_rejected():
    r = client.post("/sentence-metrics", json={"text": "woord " * 400})
    assert r.status_code == 422


def test_analysis_failure_is_a_422_not_a_crash(monkeypatch):
    from lint_ii.llm.suggestions import SuggestionEngine

    monkeypatch.setattr(SuggestionEngine, "_analyze_suggested_text", staticmethod(lambda t: None))
    r = client.post("/sentence-metrics", json={"text": "Een gewone zin."})
    assert r.status_code == 422
