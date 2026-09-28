"""Incomplete results are reported, and never cached.

Every suggestion pass is fail-open: a failed provider call (429, 5xx, timeout)
drops that call's suggestions and the analysis carries on. Until 2026-09-27
nothing recorded that, so the API cached the incomplete result and served it
for the same text until the next deploy. These tests pin the three links:
the wrapper counts, the engine reports the count, the API refuses to cache.
No test makes a real LLM call.
"""

import os
import threading
from concurrent.futures import Future

import pytest

from lint_ii import ReadabilityAnalysis
from lint_ii.llm.providers import LLMResponse
from lint_ii.llm.suggestions import SuggestionEngine, _FailureCountingProvider

# Long enough to trigger several passes (spelling, word frequency, a rewrite).
TEXT = (
    "Het college van burgemeester en wethouders heeft na uitgebreide consultatie "
    "van belanghebbenden en het inwinnen van juridisch advies besloten dat de "
    "subsidieverordening met terugwerkende kracht wordt aangepast, waardoor "
    "aanvragen die na de peildatum zijn ingediend opnieuw beoordeeld moeten worden. "
    "De implementatie geschiedt gefaseerd."
)


class _Provider:
    """Duck-typed provider: fails every call, or answers every call empty."""

    supports_concurrency = True

    def __init__(self, fail: bool):
        self.fail = fail
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def model_name(self):
        return "fake"

    def complete(self, prompt, system_prompt=None, max_tokens=None):
        with self._lock:
            self.calls += 1
        if self.fail:
            raise RuntimeError("429 Too Many Requests")
        return LLMResponse(content="", model="fake")


class TestWrapper:
    def test_counts_and_reraises(self):
        w = _FailureCountingProvider(_Provider(fail=True))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                w.complete("p")
        assert w.failures == 3

    def test_success_is_not_counted(self):
        w = _FailureCountingProvider(_Provider(fail=False))
        assert w.complete("p").content == ""
        assert w.failures == 0

    def test_delegates_attributes(self):
        w = _FailureCountingProvider(_Provider(fail=False))
        assert w.model_name == "fake"
        assert w.supports_concurrency is True

    def test_counts_exactly_under_threads(self):
        # One analysis runs its jobs on a thread pool.
        w = _FailureCountingProvider(_Provider(fail=True))

        def hammer():
            for _ in range(200):
                try:
                    w.complete("p")
                except RuntimeError:
                    pass

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert w.failures == 1600


class TestEngineReports:
    def _run(self, provider):
        engine = SuggestionEngine(provider=provider)
        analysis = ReadabilityAnalysis.from_text(TEXT)
        return engine.generate_suggestions(analysis)

    def test_failed_calls_are_reported_not_raised(self):
        provider = _Provider(fail=True)
        result = self._run(provider)  # fail-open: the analysis itself succeeds
        assert provider.calls > 0
        assert result.provider_failures == provider.calls
        assert result.as_dict()["provider_failures"] == provider.calls

    def test_clean_run_reports_zero(self):
        provider = _Provider(fail=False)
        result = self._run(provider)
        assert provider.calls > 0
        assert result.provider_failures == 0

    def test_count_is_per_analysis(self):
        # The service shares one provider; each analysis must start from zero.
        provider = _Provider(fail=True)
        assert self._run(provider).provider_failures > 0
        provider.fail = False
        assert self._run(provider).provider_failures == 0


class TestApiDoesNotCacheIncomplete:
    @pytest.fixture
    def api(self, monkeypatch):
        pytest.importorskip("fastapi")
        pytest.importorskip("langid")
        os.environ.setdefault("LINT_PROVIDER", "ollama")
        import api as api_module

        stored = {}
        monkeypatch.setattr(api_module, "_cache_put", lambda k, r: stored.__setitem__(k, r))
        api_module._stored = stored
        return api_module

    @staticmethod
    def _done(result):
        f = Future()
        f.set_result(result)
        return f

    def test_incomplete_result_is_served_but_not_cached(self, api):
        result = {"suggestions": {"suggestions": [], "provider_failures": 2}}
        api._jobs["job-incomplete"] = {"status": "pending"}  # as /analyze registers it
        api._store_job_result("job-incomplete", "key-incomplete", self._done(result))
        assert "key-incomplete" not in api._stored
        # The requesting client still gets its (partial) result.
        assert api._jobs["job-incomplete"]["status"] == "done"

    def test_complete_result_is_cached(self, api):
        result = {"suggestions": {"suggestions": [], "provider_failures": 0}}
        api._store_job_result("job-complete", "key-complete", self._done(result))
        assert "key-complete" in api._stored

    def test_cacheable(self, api):
        assert api._cacheable({"suggestions": {"provider_failures": 0}})
        assert not api._cacheable({"suggestions": {"provider_failures": 1}})
        # Results from before the field existed carry no count.
        assert api._cacheable({"suggestions": {}})
